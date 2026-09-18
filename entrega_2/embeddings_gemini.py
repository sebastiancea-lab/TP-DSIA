"""Embeddings de Gemini compartidos por A.4 (FAISS), B.1-B.4 (ChromaDB) y B.5 (ETL).

Un solo modelo y una sola dimension para toda la entrega: el indice FAISS, la coleccion
de ChromaDB y la purga semantica comparan vectores del mismo espacio.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

DIRECTORIO_ENTREGA = Path(__file__).resolve().parent
DIRECTORIO_PROYECTO = DIRECTORIO_ENTREGA.parent


def cargar_configuracion_embeddings() -> tuple[str, int]:
    load_dotenv(DIRECTORIO_PROYECTO / ".env")
    modelo = os.getenv("EMBEDDING_MODEL")
    dimension_texto = os.getenv("EMBEDDING_DIMENSION")
    if not modelo or not dimension_texto:
        raise RuntimeError("Faltan EMBEDDING_MODEL o EMBEDDING_DIMENSION en .env.")
    try:
        dimension = int(dimension_texto)
    except ValueError as exc:
        raise ValueError("EMBEDDING_DIMENSION debe ser un entero positivo.") from exc
    if dimension <= 0:
        raise ValueError("EMBEDDING_DIMENSION debe ser un entero positivo.")
    return modelo, dimension


MODELO_EMBEDDINGS, DIMENSION_EMBEDDINGS = cargar_configuracion_embeddings()


def crear_cliente() -> genai.Client:
    clave = os.getenv("LLM_API_KEY")
    if not clave:
        raise RuntimeError("Falta LLM_API_KEY en el archivo .env del proyecto.")
    return genai.Client(api_key=clave)


def obtener_embeddings(cliente: genai.Client, textos: list[str]) -> np.ndarray:
    respuesta = cliente.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=textos,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSION_EMBEDDINGS),
    )
    vectores = np.asarray([item.values for item in respuesta.embeddings], dtype=np.float32)
    if vectores.shape != (len(textos), DIMENSION_EMBEDDINGS):
        raise ValueError(f"Dimensiones inesperadas en los embeddings: {vectores.shape}.")
    return vectores
