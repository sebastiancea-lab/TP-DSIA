"""Entrega 3: Pipeline RAG Unificado de Portalia (Partes A y B).

Soporta:
  --modo basico   : RAG elemental de Parte A (k=3, colección original portalia_knowledge)
  --modo avanzado : RAG avanzado de Parte B (k=8 candidatos, chunking con overlap, LLM Reranker)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from entrega_3.parte_a import rag_pipeline as pipeline_a
from entrega_3.parte_b import rag_avanzado as pipeline_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consulta", nargs="?", help="Pregunta del cliente a evaluar.")
    parser.add_argument(
        "--modo",
        choices=["basico", "avanzado"],
        default="avanzado",
        help="Selecciona la arquitectura de RAG (por defecto: avanzado).",
    )
    parser.add_argument("--k", type=int, default=None, help="Cantidad de documentos a recuperar.")
    parser.add_argument("--reindexar", action="store_true", help="Reindexa los chunks de la Parte B.")
    opciones = parser.parse_args()

    if opciones.reindexar:
        pipeline_b.indexar_con_chunking()
        if not opciones.consulta:
            return

    consulta = opciones.consulta or (
        "Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y "
        "quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata"
    )

    if opciones.modo == "basico":
        k_val = opciones.k or 3
        print(f"\n=== EJECUTANDO RAG BÁSICO (Parte A) [k={k_val}] ===")
        pipe_basico = pipeline_a.crear_pipeline(k=k_val)
        resultado = pipe_basico.invoke(consulta)
        salida = {
            "modo": "basico",
            "pregunta": resultado["pregunta"],
            "documentos_recuperados": [
                {"id": d.id, "texto": d.page_content, "metadatos": d.metadata}
                for d in resultado["documentos"]
            ],
            "respuesta": resultado["respuesta"],
        }
        print(json.dumps(salida, ensure_ascii=False, indent=2))
    else:
        k_cand = opciones.k or pipeline_b.K_CANDIDATOS
        print(f"\n=== EJECUTANDO RAG AVANZADO (Parte B) [k_candidatos={k_cand}, LLM Reranker] ===")
        pipe_avanzado = pipeline_b.crear_pipeline_avanzado(k_candidatos=k_cand)
        resultado = pipe_avanzado.invoke(consulta)
        salida = {
            "modo": "avanzado",
            "pregunta": resultado["pregunta"],
            "candidatos_recuperados_k": len(resultado["ids_candidatos"]),
            "ids_candidatos": resultado["ids_candidatos"],
            "seleccionados_rerank_k": len(resultado["ids_seleccionados"]),
            "ids_seleccionados": resultado["ids_seleccionados"],
            "documentos_finales": [
                {"id": d.id, "doc_origen": d.metadata.get("doc_origen_id"), "texto": d.page_content}
                for d in resultado["documentos"]
            ],
            "respuesta": resultado["respuesta"],
        }
        print(json.dumps(salida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
