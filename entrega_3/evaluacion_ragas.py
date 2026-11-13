"""Entrega 3, Parte C: golden dataset (C.1) y evaluación RAGAS básico vs. avanzado (C.2, C.3)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from entrega_3.parte_a import rag_pipeline as rag_a

QUI = Path(__file__).resolve().parent
DIR_PARCIALES = QUI / "resultados_ragas"
K_BASICO = 3  # misma configuración con la que se documentó la falla de B.1
METRICAS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DISTRIBUCION = {"simple": 5, "compleja": 5, "escape": 2, "informal": 3}
URL_GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai/"

# C.1 — Golden dataset. Las respuestas esperadas salen del texto de
# entrega_2/base_conocimiento_limpia.json; en las de escape el esperado es la frase de escape.
GOLDEN_DATASET = [
    {
        "id": "S1", "tipo": "simple",
        "pregunta": "¿Cuántos días tengo para pedir un cambio de producto?",
        "respuesta_esperada": "Se puede solicitar un cambio dentro de los 30 días corridos posteriores a la recepción del pedido.",
        "docs_referencia": ["POL-CAM-001"],
    },
    {
        "id": "S2", "tipo": "simple",
        "pregunta": "Mi pedido figura como despachado, ¿me llega hoy?",
        "respuesta_esperada": "No necesariamente. Despachado significa que salió de la preparación y fue entregado al circuito logístico, pero no implica que esté en reparto ni que llegue ese mismo día; para una fecha actualizada se consulta el seguimiento vigente del envío.",
        "docs_referencia": ["ENV-003"],
    },
    {
        "id": "S3", "tipo": "simple",
        "pregunta": "¿Qué datos tengo que dar para iniciar una devolución?",
        "respuesta_esperada": "Hay que informar el número de pedido, el artículo que se desea devolver y el motivo de la solicitud.",
        "docs_referencia": ["FAQ-002"],
    },
    {
        "id": "C1", "tipo": "compleja",
        "pregunta": "Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata",
        "respuesta_esperada": "El cambio es posible porque el plazo es de 30 días corridos desde la recepción; por otro producto se verifican las condiciones y la disponibilidad, y se informa la diferencia de precio y cómo se gestiona antes de registrar nada. La devolución se puede solicitar solo dentro de los 10 días corridos posteriores a la recepción, por lo que a los 15 días ya no corresponde.",
        "docs_referencia": ["POL-CAM-001", "POL-CAM-004", "POL-DEV-001"],
    },
    {
        "id": "C2", "tipo": "compleja",
        "pregunta": "Me aceptaron la devolución, ¿qué verificaron y ya tengo el reintegro?",
        "respuesta_esperada": "Portalia verifica que la solicitud se haya iniciado dentro de los 10 días corridos de la recepción, que el producto corresponda al pedido y el estado del artículo y sus accesorios. Que la devolución esté aceptada no significa que el dinero ya se haya reintegrado: el reintegro se inicia tras completar las verificaciones y su estado se consulta en el sistema de pagos o postventa, sin prometer fecha.",
        "docs_referencia": ["POL-DEV-002", "POL-DEV-003"],
    },
    {
        "id": "C3", "tipo": "compleja",
        "pregunta": "La notebook me llegó con la caja mojada y además no prende, ¿qué hago?",
        "respuesta_esperada": "Se puede informar el número de pedido, el equipo, los síntomas y desde cuándo ocurren, y Portalia puede pedir fotografías del producto, los accesorios y el paquete para documentar el estado. La evidencia se vincula al pedido y se deriva a evaluación de postventa y garantía; no se atribuye el daño al cliente ni se promete reparación, cambio o reintegro antes de la revisión.",
        "docs_referencia": ["POL-GAR-001", "POL-GAR-002"],
    },
    {
        "id": "E1", "tipo": "escape",
        "pregunta": "¿Aceptan pago en cuotas con tarjeta de crédito?",
        "respuesta_esperada": rag_a.FRASE_ESCAPE,
        "docs_referencia": [],
    },
    {
        "id": "E2", "tipo": "escape",
        "pregunta": "¿Cuánto cuesta el televisor Vista 50 QLED?",
        "respuesta_esperada": rag_a.FRASE_ESCAPE,
        "docs_referencia": [],
    },
    {
        "id": "I1", "tipo": "informal",
        "pregunta": "che el auri se me rompió a la semana, me lo cambian o q onda",
        "respuesta_esperada": "Se puede informar el número de pedido, el equipo afectado, los síntomas y desde cuándo ocurren; Portalia verifica la compra y las condiciones de garantía antes de indicar el trámite, sin prometer reparación, cambio ni reintegro antes de la evaluación.",
        "docs_referencia": ["POL-GAR-001"],
    },
    {
        "id": "I2", "tipo": "informal",
        "pregunta": "quiero el mismo celu pero en otro color, se puede?",
        "respuesta_esperada": "Sí, se puede solicitar el cambio por otra variante (color o capacidad) del mismo modelo. Antes de confirmar, Portalia verifica las condiciones de cambio y consulta el inventario actualizado para ver si la variante elegida está disponible.",
        "docs_referencia": ["POL-CAM-003"],
    },
    {
        "id": "S4", "tipo": "simple",
        "pregunta": "¿El monitor Detail 27 USB-C permite transmitir video y cargar una notebook con un solo cable? ¿Qué hay que verificar?",
        "respuesta_esperada": "Sí, permite transportar video y suministrar energía a una notebook compatible mediante un solo cable USB-C. Antes de recomendar la función de carga se debe verificar la potencia requerida por el equipo conectado.",
        "docs_referencia": ["PROD-018"],
    },
    {
        "id": "S5", "tipo": "simple",
        "pregunta": "¿Cuándo se deriva una consulta a un operador humano y qué información debe acompañarla?",
        "respuesta_esperada": "Se deriva cuando no se puede identificar con seguridad el pedido o la intención, falta información indispensable que el cliente no puede aportar, o se requiere una decisión manual. La derivación debe incluir el motivo, los datos ya verificados y el contexto útil de la conversación, sin afirmar que la solicitud fue aprobada.",
        "docs_referencia": ["PROC-001"],
    },
    {
        "id": "C4", "tipo": "compleja",
        "pregunta": "Quiero cambiar por otro producto, pero todavía no me informaron la diferencia de precio ni acepté los términos. ¿Pueden registrar el cambio igual?",
        "respuesta_esperada": "No. Antes de registrar el cambio deben verificarse la identidad del pedido, el artículo, las condiciones aplicables, la disponibilidad y los precios vigentes; se debe informar si existe una diferencia de precio y cómo se gestionará. Mientras falte esa información o la aceptación del cliente, no debe ejecutarse la operación: el siguiente paso es completar las verificaciones, informar los términos y obtener su confirmación.",
        "docs_referencia": ["POL-CAM-004", "PROC-002"],
    },
    {
        "id": "C5", "tipo": "compleja",
        "pregunta": "El Nova 5G 256 figura sin stock de demostración. ¿Eso significa que hoy no puedo comprarlo? ¿Cómo se confirma?",
        "respuesta_esperada": "El Nova 5G 256 tiene el indicador de stock de demostración en falso, pero ese indicador no confirma ni descarta la disponibilidad real de hoy. Para confirmarla debe consultarse el inventario actualizado del modelo y la variante solicitados. La documentación describe el procedimiento y no informa el stock actual; no permite afirmar que esa consulta ya se realizó.",
        "docs_referencia": ["PROD-003", "FAQ-003", "PROC-003"],
    },
    {
        "id": "I3", "tipo": "informal",
        "pregunta": "che, ya pasó la fecha de entrega y el correo no actualiza nada, ¿dónde está el paquete y cuándo cae?",
        "respuesta_esperada": "Podés indicar el número de pedido para consultar el backend y la última actualización de la empresa logística. Si no hay una ubicación, causa o nueva fecha confirmadas, debe indicarse esa falta de confirmación sin inventar dónde está el paquete ni cuándo llegará. La documentación explica cómo consultar el seguimiento, pero no contiene la ubicación ni una fecha de entrega de ese pedido.",
        "docs_referencia": ["ENV-001", "ENV-002"],
    },
]


def validar_dataset(dataset: list[dict]) -> None:
    conteo = {t: 0 for t in DISTRIBUCION}
    for caso in dataset:
        conteo[caso["tipo"]] = conteo.get(caso["tipo"], 0) + 1
    if conteo != DISTRIBUCION:
        raise ValueError(f"Distribución {conteo}, se esperaba {DISTRIBUCION}.")
    ids = [c["id"] for c in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay ids repetidos en el golden dataset.")


def es_escape(caso: dict) -> bool:
    return caso["tipo"] == "escape"


def clasificar_abstencion(respuesta: str) -> dict[str, bool]:
    exacta = respuesta == rag_a.FRASE_ESCAPE
    parcial = not exacta and rag_a.FRASE_ESCAPE in respuesta
    return {"se_abstuvo": exacta, "coincidencia_parcial_escape": parcial}


def huella_base(ruta: Path) -> str | None:
    """Fingerprint de documentos, metadatos y configuración, sin abrir Chroma."""
    if not ruta.is_file():
        return None
    conexion = sqlite3.connect(ruta.resolve().as_uri() + "?mode=ro", uri=True)
    huella = hashlib.sha256()
    try:
        conexion.execute("BEGIN")
        for tabla in ("collections", "collection_metadata", "embeddings", "embedding_metadata"):
            filas = conexion.execute(f"SELECT * FROM {tabla}").fetchall()
            contenido = json.dumps(sorted(filas, key=repr), ensure_ascii=False,
                                   default=lambda valor: valor.hex())
            huella.update((tabla + contenido).encode("utf-8"))
    finally:
        conexion.close()
    return huella.hexdigest()


def crear_manifiesto(nombres: list[str]) -> dict:
    """Identifica datos, código, configuración y bases usados por un experimento."""
    from dotenv import dotenv_values

    archivos = (
        "entrega_3/evaluacion_ragas.py", "entrega_3/parte_a/rag_pipeline.py",
        "entrega_3/parte_b/rag_avanzado.py", "entrega_1/parte_c/app.py",
        "entrega_1/parte_c/schemas.py", "entrega_2/embeddings_gemini.py",
        "entrega_2/vector_db.py", "entrega_2/base_conocimiento_limpia.json", "uv.lock",
    )
    configuracion = {**dotenv_values(RAIZ / ".env"), **os.environ}
    paquetes = {}
    for nombre in ("ragas", "chromadb", "google-genai", "langchain-core", "langchain-text-splitters"):
        try:
            paquetes[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            paquetes[nombre] = None
    rutas_bases = {
        "basico": RAIZ / "entrega_2/chroma_db/chroma.sqlite3",
        "avanzado": QUI / "parte_b/chroma_db/chroma.sqlite3",
    }
    return {
        "version": 1,
        "pipelines": nombres,
        "dataset": GOLDEN_DATASET,
        "archivos_sha256": {
            nombre: hashlib.sha256((RAIZ / nombre).read_bytes()).hexdigest() for nombre in archivos
        },
        # Nunca se guardan claves ni variables de telemetría.
        "configuracion": {nombre: configuracion.get(nombre) for nombre in (
            "LLM_MODEL", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
        )},
        "paquetes": paquetes,
        "python": sys.version.split()[0],
        "bases_sha256": {nombre: huella_base(rutas_bases[nombre]) for nombre in nombres},
    }


def preparar_experimento(nombre: str, retomar: bool, manifiesto: dict) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", nombre):
        raise ValueError("El experimento debe tener entre 1 y 80 letras, números, guiones o guiones bajos.")
    directorio = DIR_PARCIALES / nombre
    ruta_manifiesto = directorio / "manifiesto.json"
    if retomar:
        if not ruta_manifiesto.is_file():
            raise ValueError("No existe ese experimento con manifiesto; no se retoman parciales antiguos.")
        if json.loads(ruta_manifiesto.read_text(encoding="utf-8")) != manifiesto:
            raise ValueError("El dataset, código, configuración o base cambió. Iniciá otro --experimento.")
        if (directorio / "resultados.json").exists():
            raise ValueError("El experimento ya tiene resultados finales; iniciá otro --experimento.")
    else:
        try:
            directorio.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("El experimento ya existe. Usá --resume o elegí otro --experimento.") from exc
        with ruta_manifiesto.open("x", encoding="utf-8") as archivo:
            json.dump(manifiesto, archivo, ensure_ascii=False, indent=2)
    return directorio


def cargar_parciales(ruta: Path) -> list[dict]:
    """Rechaza resultados incompatibles o incompletos antes de llamar al proveedor."""
    if not ruta.exists():
        return []
    try:
        filas = [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
        casos = {caso["id"]: caso for caso in GOLDEN_DATASET}
        vistos = set()
        for fila in filas:
            caso = casos[fila["id"]]
            if fila["id"] in vistos or any(fila[campo] != caso[campo] for campo in (
                "tipo", "pregunta", "respuesta_esperada",
            )):
                raise ValueError("IDs repetidos o casos distintos del dataset.")
            abstencion = clasificar_abstencion(fila["respuesta"])
            if any(fila[clave] != valor for clave, valor in abstencion.items()) or (
                fila["abstencion_ok"] != (
                    abstencion["se_abstuvo"] == es_escape(caso)
                    and not abstencion["coincidencia_parcial_escape"]
                )
            ):
                raise ValueError("Clasificación de abstención incompatible.")
            for metrica in METRICAS:
                valor = fila[metrica]
                if valor is not None and (isinstance(valor, bool) or not isinstance(valor, (int, float))
                                          or not math.isfinite(valor) or not 0 <= valor <= 1):
                    raise ValueError("Métrica inválida.")
            if not isinstance(fila["ids_contexto"], list):
                raise ValueError("Contextos inválidos.")
            vistos.add(fila["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Parciales inválidos en {ruta.name}; no se modificaron.") from exc
    return filas


def promedio(valores: list[float]) -> tuple[float | None, int]:
    """Promedio ignorando NaN; devuelve también cuántos casos entraron."""
    validos = [v for v in valores if v is not None and not math.isnan(v)]
    return (sum(validos) / len(validos) if validos else None), len(validos)


def con_reintentos(fn, intentos: int = 4):
    """Espera y reintenta ante el límite por minuto o saturación; corta si es el límite diario."""
    for i in range(intentos):
        try:
            return fn()
        except Exception as exc:
            if "PerDay" in str(exc):
                raise rag_a.ErrorRAG("Cuota diaria de Gemini agotada; retomá mañana con --resume.") from exc
            reintentable = any(c in str(exc) for c in ("429", "RESOURCE_EXHAUSTED", "503"))
            if not reintentable or i == intentos - 1:
                raise
            print("    (Gemini pidió esperar, espero 30s...)")
            time.sleep(30)


def crear_juez():
    """Juez RAGAS: Gemini por su endpoint compatible con OpenAI (el cliente async de google-genai
    falla con RAGAS 0.4), más embeddings de Gemini para Answer Relevancy."""
    from google import genai
    from openai import AsyncOpenAI
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
    )

    app = rag_a.importar_dependencia("entrega_1.parte_c.app")
    clave, modelo = app.cargar_config()
    llm = llm_factory(modelo, client=AsyncOpenAI(api_key=clave, base_url=URL_GEMINI_OPENAI))
    embeddings = GoogleEmbeddings(client=genai.Client(api_key=clave), model="gemini-embedding-001")
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def medir(juez: dict, caso: dict, respuesta: str, contextos: list[str]) -> dict[str, float]:
    q, ref = caso["pregunta"], caso["respuesta_esperada"]
    llamadas = {
        "faithfulness": lambda: juez["faithfulness"].score(
            user_input=q, response=respuesta, retrieved_contexts=contextos),
        "answer_relevancy": lambda: juez["answer_relevancy"].score(user_input=q, response=respuesta),
        "context_precision": lambda: juez["context_precision"].score(
            user_input=q, reference=ref, retrieved_contexts=contextos),
        "context_recall": lambda: juez["context_recall"].score(
            user_input=q, reference=ref, retrieved_contexts=contextos),
    }
    resultado = {}
    for nombre, fn in llamadas.items():
        # Sin contextos las métricas de contexto no se pueden calcular: quedan en NaN.
        if not contextos and nombre != "answer_relevancy":
            resultado[nombre] = float("nan")
            continue
        resultado[nombre] = float(con_reintentos(fn).value)
    return resultado


def crear_pipeline(nombre: str):
    if nombre == "basico":
        return rag_a.crear_pipeline(K_BASICO)
    from entrega_3.parte_b import rag_avanzado
    return rag_avanzado.crear_pipeline_avanzado()


def evaluar_pipeline(nombre: str, juez: dict, retomar: bool, directorio: Path) -> list[dict]:
    ruta = directorio / f"parcial_{nombre}.jsonl"
    if retomar:
        filas = cargar_parciales(ruta)
        print(f"[{nombre}] retomando: {len(filas)} casos ya medidos.")
    else:
        with ruta.open("x", encoding="utf-8"):
            pass
        filas = []
    hechos = {f["id"] for f in filas}
    if hechos == {caso["id"] for caso in GOLDEN_DATASET}:
        return filas
    cadena = crear_pipeline(nombre)
    for caso in GOLDEN_DATASET:
        if caso["id"] in hechos:
            continue
        print(f"[{nombre}] {caso['id']}: {caso['pregunta'][:70]}")
        salida = con_reintentos(lambda: cadena.invoke(caso["pregunta"]))
        respuesta = salida["respuesta"]
        contextos = [d.page_content for d in salida["documentos"]]
        abstencion = clasificar_abstencion(respuesta)
        if es_escape(caso):
            # Fuera de dominio: lo que importa es la abstención, las métricas no aplican.
            puntajes = {m: float("nan") for m in METRICAS}
        else:
            puntajes = medir(juez, caso, respuesta, contextos)
        fila = {
            "id": caso["id"], "tipo": caso["tipo"], "pregunta": caso["pregunta"],
            "respuesta": respuesta, "respuesta_esperada": caso["respuesta_esperada"],
            "ids_contexto": [d.id for d in salida["documentos"]],
            **abstencion,
            "abstencion_ok": (abstencion["se_abstuvo"] == es_escape(caso)
                              and not abstencion["coincidencia_parcial_escape"]),
            **{m: (None if math.isnan(v) else v) for m, v in puntajes.items()},
        }
        filas.append(fila)
        # Un JSONL válido puede terminar sin salto de línea tras una interrupción.
        # Al retomar, el parcial de este pipeline puede no existir todavía.
        necesita_salto = (ruta.exists() and ruta.stat().st_size > 0
                          and not ruta.read_bytes().endswith(b"\n"))
        with ruta.open("a", encoding="utf-8") as f:
            if necesita_salto:
                f.write("\n")
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print("    " + " | ".join(f"{m}={fila[m]}" if fila[m] is None else f"{m}={fila[m]:.2f}" for m in METRICAS)
              + f" | abstencion_ok={fila['abstencion_ok']}")
    return filas


def resumir(filas: list[dict]) -> dict:
    resumen = {}
    for m in METRICAS:
        valor, n = promedio([f[m] for f in filas])
        resumen[m] = {"promedio": valor, "casos": n}
    resumen["abstenciones_correctas"] = sum(f["abstencion_ok"] for f in filas)
    resumen["total_casos"] = len(filas)
    return resumen


def tabla_comparativa(resumenes: dict) -> str:
    if set(resumenes) != {"basico", "avanzado"}:
        return ""
    fmt = lambda v: "n/d" if v is None else f"{v:.3f}"
    lineas = ["| Métrica | Básico | Avanzado | Delta |", "|---|---|---|---|"]
    for m in METRICAS:
        b, a = resumenes["basico"][m], resumenes["avanzado"][m]
        delta = "n/d" if b["promedio"] is None or a["promedio"] is None else f"{a['promedio'] - b['promedio']:+.3f}"
        lineas.append(f"| {m} | {fmt(b['promedio'])} (n={b['casos']}) | {fmt(a['promedio'])} (n={a['casos']}) | {delta} |")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("basico", "avanzado", "ambos"), default="ambos")
    parser.add_argument("--solo-dataset", action="store_true", help="Valida e imprime el dataset sin llamar a la API.")
    parser.add_argument("--resume", action="store_true", help="Retoma desde los casos ya medidos.")
    parser.add_argument("--experimento", help="Identificador de la corrida; obligatorio para --resume.")
    opciones = parser.parse_args(argv)
    try:
        validar_dataset(GOLDEN_DATASET)
        if opciones.solo_dataset:
            print(json.dumps(GOLDEN_DATASET, ensure_ascii=False, indent=2))
            print(f"\nDataset válido: {len(GOLDEN_DATASET)} preguntas {DISTRIBUCION}.")
            return 0
        nombres = ["basico", "avanzado"] if opciones.pipeline == "ambos" else [opciones.pipeline]
        if opciones.resume and not opciones.experimento:
            raise ValueError("--resume requiere --experimento para identificar la corrida.")
        nombre = opciones.experimento or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        directorio = preparar_experimento(nombre, opciones.resume, crear_manifiesto(nombres))
        # Verificar todos los parciales antes de construir un juez o pipeline.
        if opciones.resume:
            for pipeline in nombres:
                cargar_parciales(directorio / f"parcial_{pipeline}.jsonl")
        print(f"Experimento: {nombre}")
        juez = crear_juez()
        resultados = {n: evaluar_pipeline(n, juez, opciones.resume, directorio) for n in nombres}
        resumenes = {n: resumir(f) for n, f in resultados.items()}
        # Creación exclusiva: nunca se sobrescribe la corrida histórica ni una finalizada.
        salida_json = directorio / "resultados.json"
        with salida_json.open("x", encoding="utf-8") as archivo:
            json.dump({"experimento": nombre, "resumen": resumenes, "casos": resultados},
                      archivo, ensure_ascii=False, indent=2)
    except (rag_a.ErrorRAG, ValueError) as exc:
        print(f"[ragas] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ragas] Falló la evaluación ({type(exc).__name__}): {str(exc)[:200]}", file=sys.stderr)
        return 1
    print(f"\nResultados en {salida_json}")
    print(tabla_comparativa(resumenes) or json.dumps(resumenes, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
