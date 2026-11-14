"""Entrega 4, Fase 4: matriz de validación del agente (evaluación de trayectoria).

Hace llamadas REALES a Gemini (pocas, por la cuota del plan gratuito). Cada caso corre
en un thread nuevo y con una base de memoria y de solicitudes temporal, para no
ensuciar los datos reales. Sale con exit code 1 si falla tool_selection o sin_prohibidas.

Uso: uv run python entrega_4/validacion_agente.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from entrega_4 import config  # noqa: E402,F401  (load_dotenv primero)
from entrega_3.parte_a.rag_pipeline import ErrorRAG  # noqa: E402
from entrega_4.agente import (  # noqa: E402
    ErrorDeConfig, config_invocacion, crear_agente, entrada_turno,
)
from entrega_4.backend import BackendPortalia  # noqa: E402
from entrega_4.errores import ERRORES_EXTERNOS  # noqa: E402
from entrega_4.memoria import crear_saver  # noqa: E402
from entrega_4.observabilidad import (  # noqa: E402
    herramientas_del_turno, mensajes_del_turno, tokens_del_turno,
)
from entrega_4.tools import TOOL_ACTUADORA  # noqa: E402

REINTENTOS_429 = 3
ESPERA_429_SEG = 30


@dataclass(frozen=True)
class Escenario:
    id: str
    nombre: str
    pregunta: str
    esperadas: list[str]                       # subsecuencia ordenada de tools
    prohibidas: list[str]                      # tools que NO pueden aparecer
    contiene: list[str] = field(default_factory=list)  # alguno debe estar en la respuesta


ESCENARIOS = [
    Escenario(
        "1", "Consulta de política",
        "¿Cuántos días tengo para devolver un producto?",
        ["buscar_politicas_postventa"], [TOOL_ACTUADORA], ["10"],
    ),
    Escenario(
        "2", "Estado de pedido",
        "¿Dónde está mi pedido 48327?",
        ["consultar_pedido"], [TOOL_ACTUADORA], ["despachado", "en camino", "camino"],
    ),
    Escenario(
        "3", "Cambio elegible",
        "Quiero cambiar los auriculares del pedido 51204 por el color blanco, porque me "
        "gusta más. Confirmo que quiero iniciar el cambio ahora.",
        ["consultar_pedido", TOOL_ACTUADORA], [], ["SOL-"],
    ),
    Escenario(
        "4a", "Devolución vencida (inelegible)",
        "Quiero devolver el celular del pedido 50718 porque no me convence. Confirmo, "
        "iniciá la devolución.",
        ["consultar_pedido"], [TOOL_ACTUADORA], ["plazo", "días", "dias"],
    ),
    Escenario(
        "4b", "Fuera de dominio",
        "¿Quién ganó el Mundial de fútbol de 2022?",
        [], [TOOL_ACTUADORA, "consultar_pedido"], ["no poseo información", "postventa"],
    ),
]


def es_subsecuencia(esperadas: list[str], usadas: list[str]) -> bool:
    iterador = iter(usadas)
    return all(nombre in iterador for nombre in esperadas)


def correr_escenario(grafo, escenario: Escenario) -> dict:
    thread_id = f"val-{escenario.id}-{uuid.uuid4().hex[:8]}"
    for intento in range(1, REINTENTOS_429 + 1):
        try:
            resultado = grafo.invoke(
                entrada_turno(escenario.pregunta), config_invocacion(thread_id),
            )
            break
        except ERRORES_EXTERNOS as exc:  # solo para reintentar cuota; el resto se relanza
            es_cuota = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if not es_cuota or intento == REINTENTOS_429:
                raise
            print(f"  [cuota] 429 en {escenario.id}, espero {ESPERA_429_SEG}s...")
            time.sleep(ESPERA_429_SEG)
    nuevos = mensajes_del_turno(resultado["messages"])
    usadas = herramientas_del_turno(nuevos)
    respuesta = (nuevos[-1].text or "").strip() if nuevos else ""
    entrada, salida = tokens_del_turno(nuevos)
    return {
        "escenario": escenario, "usadas": usadas, "respuesta": respuesta,
        "tokens_entrada": entrada, "tokens_salida": salida,
        "tool_selection": es_subsecuencia(escenario.esperadas, usadas),
        "sin_prohibidas": not set(escenario.prohibidas) & set(usadas),
        "contiene": (
            not escenario.contiene
            or any(t.lower() in respuesta.lower() for t in escenario.contiene)
        ),
    }


def marca(ok: bool) -> str:
    return "OK " if ok else "FALLA"


def main() -> int:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            saver = crear_saver(tmp / "memoria.sqlite3")
            backend = BackendPortalia(ruta_solicitudes=tmp / "solicitudes.sqlite3")
            grafo = crear_agente(saver, backend=backend)
            resultados = []
            for escenario in ESCENARIOS:
                print(f"> {escenario.id} {escenario.nombre}: {escenario.pregunta}")
                resultados.append(correr_escenario(grafo, escenario))
            saver.conn.close()
    except (ErrorDeConfig, ErrorRAG) as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2

    print(f"\n{'#':<3} {'Escenario':<32} {'tool_sel':<9} {'sin_proh':<9} {'contiene':<9} "
          f"{'tokens in/out':<14} Trayectoria")
    for r in resultados:
        e = r["escenario"]
        print(f"{e.id:<3} {e.nombre:<32} {marca(r['tool_selection']):<9} "
              f"{marca(r['sin_prohibidas']):<9} {marca(r['contiene']):<9} "
              f"{r['tokens_entrada']}/{r['tokens_salida']:<10} {' -> '.join(r['usadas']) or '(sin tools)'}")
    print()
    for r in resultados:
        print(f"[{r['escenario'].id}] {r['respuesta']}\n")

    fallas = [r for r in resultados if not (r["tool_selection"] and r["sin_prohibidas"])]
    avisos = [r for r in resultados if not r["contiene"]]
    if avisos:
        print("Aviso: 'contiene' no se cumplió en", ", ".join(r["escenario"].id for r in avisos),
              "(revisar la redacción; no bloquea).")
    if fallas:
        print("FALLARON tool_selection / sin_prohibidas en:",
              ", ".join(r["escenario"].id for r in fallas))
        return 1
    print("Matriz OK: trayectorias correctas en todos los escenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
