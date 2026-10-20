"""Entrega 3, Parte C: golden dataset (C.1) y evaluación RAGAS básico vs. avanzado (C.2, C.3)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from entrega_3.parte_a import rag_pipeline as rag_a

QUI = Path(__file__).resolve().parent
SALIDA_JSON = QUI / "resultados_ragas.json"
DIR_PARCIALES = QUI / "resultados_ragas"
K_BASICO = 3  # misma configuración con la que se documentó la falla de B.1
METRICAS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DISTRIBUCION = {"simple": 3, "compleja": 3, "escape": 2, "informal": 2}
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


def evaluar_pipeline(nombre: str, juez: dict, retomar: bool) -> list[dict]:
    DIR_PARCIALES.mkdir(exist_ok=True)
    ruta = DIR_PARCIALES / f"parcial_{nombre}.jsonl"
    filas = []
    if retomar and ruta.exists():
        filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[{nombre}] retomando: {len(filas)} casos ya medidos.")
    else:
        ruta.write_text("", encoding="utf-8")
    hechos = {f["id"] for f in filas}
    cadena = crear_pipeline(nombre)
    for caso in GOLDEN_DATASET:
        if caso["id"] in hechos:
            continue
        print(f"[{nombre}] {caso['id']}: {caso['pregunta'][:70]}")
        salida = con_reintentos(lambda: cadena.invoke(caso["pregunta"]))
        respuesta = salida["respuesta"]
        contextos = [d.page_content for d in salida["documentos"]]
        escapo = rag_a.FRASE_ESCAPE in respuesta
        if es_escape(caso):
            # Fuera de dominio: lo que importa es la abstención, las métricas no aplican.
            puntajes = {m: float("nan") for m in METRICAS}
        else:
            puntajes = medir(juez, caso, respuesta, contextos)
        fila = {
            "id": caso["id"], "tipo": caso["tipo"], "pregunta": caso["pregunta"],
            "respuesta": respuesta, "respuesta_esperada": caso["respuesta_esperada"],
            "ids_contexto": [d.id for d in salida["documentos"]],
            "se_abstuvo": escapo, "abstencion_ok": escapo == es_escape(caso),
            **{m: (None if math.isnan(v) else v) for m, v in puntajes.items()},
        }
        filas.append(fila)
        with ruta.open("a", encoding="utf-8") as f:
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
    opciones = parser.parse_args(argv)
    try:
        validar_dataset(GOLDEN_DATASET)
        if opciones.solo_dataset:
            print(json.dumps(GOLDEN_DATASET, ensure_ascii=False, indent=2))
            print(f"\nDataset válido: {len(GOLDEN_DATASET)} preguntas {DISTRIBUCION}.")
            return 0
        nombres = ["basico", "avanzado"] if opciones.pipeline == "ambos" else [opciones.pipeline]
        juez = crear_juez()
        resultados = {n: evaluar_pipeline(n, juez, opciones.resume) for n in nombres}
    except (rag_a.ErrorRAG, ValueError) as exc:
        print(f"[ragas] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ragas] Falló la evaluación ({type(exc).__name__}): {str(exc)[:200]}", file=sys.stderr)
        return 1
    resumenes = {n: resumir(f) for n, f in resultados.items()}
    SALIDA_JSON.write_text(json.dumps({"resumen": resumenes, "casos": resultados},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados en {SALIDA_JSON.relative_to(RAIZ)}")
    print(tabla_comparativa(resumenes) or json.dumps(resumenes, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
