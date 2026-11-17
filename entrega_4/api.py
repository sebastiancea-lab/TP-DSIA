"""Entrega 4, Fase 7: API REST del agente Portalia.

Levantar con: uv run uvicorn entrega_4.api:app --port 8000
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from entrega_4 import config  # noqa: F401  (load_dotenv antes de langchain)
from fastapi import FastAPI, HTTPException, Path as RutaParam, Request
from langchain_core.messages import AIMessage, HumanMessage

from entrega_3.parte_a.rag_pipeline import FRASE_ESCAPE, ErrorRAG
from entrega_4.agente import (
    GraphRecursionError, config_invocacion, crear_agente, crear_llm, entrada_turno,
    mensajes_de_cierre,
)
from entrega_4.backend import BackendPortalia
from entrega_4.config import DB_PATH, LLM_MODEL, MAX_ITERACIONES
from entrega_4.errores import ERRORES_EXTERNOS
from entrega_4.guardrails import OUTPUT_PII, enmascarar_pii, validar_input
from entrega_4.memoria import crear_saver
from entrega_4.observabilidad import (
    Metricas, herramientas_del_turno, log_evento, mensajes_del_turno, obtener_logger,
    tokens_del_turno,
)
from entrega_4.prompts import MENSAJE_CORTE_ITERACIONES, MENSAJE_ERROR_TECNICO
from entrega_4.schemas import (
    PATRON_THREAD_ID, ConsultaRequest, ConsultaResponse, EstadoResponse, HealthResponse,
    MensajeHistorial, MetricasResponse,
)
from entrega_4.tools import crear_retriever_politicas


class RetrieverNoDisponible:
    """Reemplaza al retriever si Chroma no abre: el servidor sigue arriba y /health lo informa."""

    def __init__(self, motivo: str) -> None:
        self.motivo = motivo

    def invoke(self, consulta: str, *args, **kwargs):
        # Mantiene explícito el fallo de recuperación si una herramienta intenta
        # buscar políticas mientras Chroma no está disponible.
        raise ErrorRAG(self.motivo)


def texto_de(mensaje) -> str:
    """Texto plano de un mensaje (Gemini puede devolver el contenido en bloques)."""
    # La API devuelve texto al cliente aunque el SDK represente internamente
    # el contenido como bloques.
    return (mensaje.text or "").strip()


def crear_app(
    llm=None, ruta_db: Optional[Path] = None, backend: Optional[BackendPortalia] = None,
    retriever=None, max_iteraciones: int = MAX_ITERACIONES,
) -> FastAPI:
    """Fábrica de la app. Los parámetros permiten inyectar dobles en los tests."""

    def cerrar_turno_fallido(grafo, conf, texto: str) -> None:
        """Deja la memoria coherente tras un turno cortado: sin humano huérfano ni tool_calls abiertos."""
        # Si una invocacion termina por límite o error, guarda una respuesta de
        # cierre para que el siguiente turno no herede llamadas de herramienta pendientes.
        mensajes = grafo.get_state(conf).values.get("messages", [])
        grafo.update_state(
            conf, {"messages": mensajes_de_cierre(mensajes, texto)}, as_node="agente",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Los recursos duraderos y el grafo se crean al arrancar el servidor,
        # y se comparten entre las solicitudes de esta instancia.
        obtener_logger()
        saver = crear_saver(ruta_db or DB_PATH)
        coleccion_ok = True
        retriever_vivo = retriever
        if retriever_vivo is None:
            try:
                retriever_vivo = crear_retriever_politicas()
            except ErrorRAG as exc:
                # La API puede iniciar en modo degradado; el endpoint de salud lo
                # informa y las búsquedas fallan explícitamente al ser invocadas.
                coleccion_ok = False
                retriever_vivo = RetrieverNoDisponible(str(exc))
                log_evento("chroma_no_disponible", logging.ERROR, motivo=str(exc))
        app.state.saver = saver
        app.state.grafo = crear_agente(
            saver, llm=llm or crear_llm(), backend=backend, retriever=retriever_vivo,
            max_iteraciones=max_iteraciones,
        )
        app.state.metricas = Metricas()
        app.state.coleccion_ok = coleccion_ok
        log_evento("inicio", modelo=LLM_MODEL, coleccion_chroma=coleccion_ok)
        yield
        saver.conn.close()

    app = FastAPI(title="Portalia - Agente de postventa", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        # Este estado refleja si la colección de políticas está disponible;
        # "degradado" no impide que la API permanezca levantada.
        ok = request.app.state.coleccion_ok
        return HealthResponse(
            estado="ok" if ok else "degradado", modelo=LLM_MODEL, coleccion_chroma=ok,
        )

    @app.post("/consultar", response_model=ConsultaResponse)
    def consultar(pedido: ConsultaRequest, request: Request) -> ConsultaResponse:
        estado = request.app.state
        inicio = time.perf_counter()
        # Reutilizar el ID continúa la conversación persistida; si no llega uno,
        # se crea un hilo nuevo y el ID se devuelve al cliente.
        thread_id = pedido.thread_id or uuid.uuid4().hex

        def latencia() -> float:
            return round((time.perf_counter() - inicio) * 1000, 2)

        # 1) Guardrail de input: si bloquea, el grafo ni se invoca.
        bloqueo = validar_input(pedido.pregunta)
        if bloqueo:
            # Un rechazo de input no llega al agente ni consume una llamada al LLM.
            estado.metricas.registrar_bloqueo(bloqueo.tipo)
            estado.metricas.registrar_consulta(latencia())
            log_evento("bloqueo_input", logging.WARNING, thread_id=thread_id,
                       bloqueo=bloqueo.tipo, latencia_ms=latencia())
            return ConsultaResponse(
                thread_id=thread_id, respuesta=bloqueo.mensaje, herramientas_usadas=[],
                bloqueado=True, motivo_bloqueo=bloqueo.tipo, latencia_ms=latencia(),
            )

        # 2) Agente, con errores separados por capa (nunca un except genérico).
        conf = config_invocacion(thread_id)
        try:
            resultado = estado.grafo.invoke(entrada_turno(pedido.pregunta.strip()), conf)
        except GraphRecursionError:
            # El nodo de cierre deja el historial consistente antes de devolver
            # el mensaje que informa que se alcanzó el límite de iteraciones.
            estado.metricas.registrar_consulta(latencia(), error=True)
            log_evento("recursion_limit", logging.ERROR, thread_id=thread_id, latencia_ms=latencia())
            cerrar_turno_fallido(estado.grafo, conf, MENSAJE_CORTE_ITERACIONES)
            return ConsultaResponse(
                thread_id=thread_id, respuesta=MENSAJE_CORTE_ITERACIONES,
                herramientas_usadas=[], bloqueado=False, latencia_ms=latencia(),
            )
        except ERRORES_EXTERNOS as exc:
            # Una falla del RAG llega al agente como error de tool (no acá): ver tools.py.
            estado.metricas.registrar_consulta(latencia(), error=True)
            log_evento("error_llm", logging.ERROR, thread_id=thread_id,
                       tipo=type(exc).__name__)
            cerrar_turno_fallido(estado.grafo, conf, MENSAJE_ERROR_TECNICO)
            raise HTTPException(502, f"Falla en el modelo de lenguaje ({type(exc).__name__}).") from exc
        except sqlite3.Error as exc:
            # La memoria o los tickets no responden: no se intenta cerrar el turno en esa misma base.
            estado.metricas.registrar_consulta(latencia(), error=True)
            log_evento("error_sqlite", logging.ERROR, thread_id=thread_id,
                       tipo=type(exc).__name__)
            raise HTTPException(500, f"Falla en el almacenamiento ({type(exc).__name__}).") from exc

        # Se separan los mensajes creados en este turno del historial completo
        # persistido para informar herramientas, tokens y respuesta recientes.
        nuevos = mensajes_del_turno(resultado["messages"])
        herramientas = herramientas_del_turno(nuevos)
        respuesta_ia = next((m for m in reversed(nuevos) if isinstance(m, AIMessage)), None)
        respuesta = texto_de(respuesta_ia) if respuesta_ia else ""
        vacia = not respuesta
        if vacia:
            # Una salida vacía no se entrega al cliente como éxito sin contenido.
            respuesta = FRASE_ESCAPE
            log_evento("respuesta_vacia", logging.WARNING, thread_id=thread_id)

        # 3) Guardrail de output: enmascara PII también en la memoria (mismo id de mensaje).
        enmascarada, tipos_pii = enmascarar_pii(respuesta)
        if tipos_pii:
            estado.metricas.registrar_bloqueo(OUTPUT_PII)
            log_evento("pii_enmascarada", logging.WARNING, thread_id=thread_id, tipos=tipos_pii)
        if tipos_pii or vacia:
            # También se reemplaza el mensaje en memoria: /estado no debe volver
            # a exponer PII que ya se ocultó en la respuesta HTTP.
            estado.grafo.update_state(
                conf,
                {"messages": [AIMessage(
                    content=enmascarada, id=respuesta_ia.id if respuesta_ia else None,
                )]},
                as_node="agente",
            )

        estado.metricas.registrar_consulta(
            latencia(), herramientas, tokens=tokens_del_turno(nuevos),
        )
        log_evento("consulta", thread_id=thread_id, latencia_ms=latencia(),
                   herramientas=herramientas, bloqueo=OUTPUT_PII if tipos_pii else None)
        return ConsultaResponse(
            thread_id=thread_id, respuesta=enmascarada, herramientas_usadas=herramientas,
            bloqueado=False, motivo_bloqueo=OUTPUT_PII if tipos_pii else None,
            latencia_ms=latencia(),
        )

    @app.get("/estado/{thread_id}", response_model=EstadoResponse)
    def estado_thread(
        request: Request, thread_id: str = RutaParam(pattern=PATRON_THREAD_ID),
    ) -> EstadoResponse:
        # Devuelve solo mensajes visibles para el usuario; se omiten tool calls
        # y respuestas internas de herramientas.
        instantanea = request.app.state.grafo.get_state({"configurable": {"thread_id": thread_id}})
        mensajes = (instantanea.values or {}).get("messages")
        if not mensajes:
            raise HTTPException(404, f"No existe el thread {thread_id}.")
        historial = []
        for mensaje in mensajes:
            if isinstance(mensaje, HumanMessage):
                historial.append(MensajeHistorial(rol="usuario", contenido=texto_de(mensaje)))
            elif isinstance(mensaje, AIMessage) and not mensaje.tool_calls and texto_de(mensaje):
                historial.append(MensajeHistorial(rol="asistente", contenido=texto_de(mensaje)))
        return EstadoResponse(
            thread_id=thread_id, cantidad_mensajes=len(historial), mensajes=historial,
        )

    @app.get("/metricas", response_model=MetricasResponse)
    def metricas(request: Request) -> MetricasResponse:
        # Expone el snapshot agregado, no los mensajes ni el contenido de cada consulta.
        return MetricasResponse(**request.app.state.metricas.snapshot())

    return app


# Uvicorn importa esta instancia como entrega_4.api:app; las pruebas pueden
# crear otra con dependencias temporales mediante crear_app().
app = crear_app()
