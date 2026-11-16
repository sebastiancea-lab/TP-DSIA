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
# Permite ejecutar este punto de entrada desde cualquier directorio y seguir
# importando los paquetes de ambas partes desde la raiz del proyecto.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Se importan las implementaciones existentes; este archivo solo selecciona y
# ejecuta una arquitectura, no duplica la logica de recuperacion o generacion.
from entrega_3.parte_a import rag_pipeline as pipeline_a
from entrega_3.parte_b import rag_avanzado as pipeline_b


def main():
    # Este CLI unifica las demos de A y B para comparar los dos enfoques con
    # una misma pregunta y elegir la cantidad de candidatos desde la terminal.
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
        # La reconstruccion consume embeddings y reemplaza la coleccion avanzada.
        # Se hace solo por solicitud explicita; sin consulta, termina al reindexar.
        pipeline_b.indexar_con_chunking()
        if not opciones.consulta:
            return

    # Si no se pasa una pregunta, ambas partes reciben el mismo caso de prueba
    # complejo para facilitar la comparacion.
    consulta = opciones.consulta or (
        "Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y "
        "quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata"
    )

    if opciones.modo == "basico":
        # Parte A: recupera directamente hasta k documentos completos de la
        # coleccion original y los pasa al generador.
        k_val = opciones.k or 3
        print(f"\n=== EJECUTANDO RAG BÁSICO (Parte A) [k={k_val}] ===")
        pipe_basico = pipeline_a.crear_pipeline(k=k_val)
        resultado = pipe_basico.invoke(consulta)
        # Se imprimen las fuentes junto con la respuesta para poder rastrear
        # qué documentos recibió el modelo.
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
        # Parte B: recupera candidatos de la coleccion fragmentada; el pipeline
        # avanzado los filtra con el juez antes de generar la respuesta final.
        k_cand = opciones.k or pipeline_b.K_CANDIDATOS
        print(f"\n=== EJECUTANDO RAG AVANZADO (Parte B) [k_candidatos={k_cand}, LLM Reranker] ===")
        pipe_avanzado = pipeline_b.crear_pipeline_avanzado(k_candidatos=k_cand)
        resultado = pipe_avanzado.invoke(consulta)
        # La salida expone candidatos y seleccion final para comparar el efecto
        # del reranking, además de conservar los textos usados en la respuesta.
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
    # Punto de entrada para usar la misma herramienta desde la linea de comandos.
    main()
