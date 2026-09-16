"""A.4: construye o recarga un índice FAISS del catálogo de Portalia."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


DIRECTORIO_ENTREGA = Path(__file__).resolve().parent
DIRECTORIO_PROYECTO = DIRECTORIO_ENTREGA.parent
ARCHIVO_DATOS = DIRECTORIO_ENTREGA / "base_conocimiento.json"
DIRECTORIO_INDICES = DIRECTORIO_ENTREGA / "indices"
ARCHIVO_INDICE = DIRECTORIO_INDICES / "portalia.index"
ARCHIVO_MANIFIESTO = DIRECTORIO_INDICES / "portalia.index.meta.json"


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
CONSULTAS_PRUEBA = (
    "Necesito un teléfono con red 5G y espacio para guardar muchos videos.",
    "Busco una pantalla grande con tecnología QLED para ver películas.",
    "Me arrepentí de la compra: ¿hasta cuándo puedo pedir una devolución?",
)


def cargar_documentos() -> list[dict]:
    documentos = json.loads(ARCHIVO_DATOS.read_text(encoding="utf-8"))
    if not documentos or any(not item.get("descripcion_semantica") for item in documentos):
        raise ValueError("La base debe contener descripciones semánticas no vacías.")
    ids = [item["id"] for item in documentos]
    if len(ids) != len(set(ids)):
        raise ValueError("La base contiene IDs duplicados.")
    return documentos


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


def datos_manifiesto(documentos: list[dict]) -> dict:
    return {
        "sha256_catalogo": hashlib.sha256(ARCHIVO_DATOS.read_bytes()).hexdigest(),
        "modelo": MODELO_EMBEDDINGS,
        "dimension": DIMENSION_EMBEDDINGS,
        "ids": [item["id"] for item in documentos],
    }


def cargar_indice_vigente(esperado: dict) -> faiss.Index | None:
    if not ARCHIVO_INDICE.exists() or not ARCHIVO_MANIFIESTO.exists():
        return None
    try:
        guardado = json.loads(ARCHIVO_MANIFIESTO.read_text(encoding="utf-8"))
        if guardado != esperado:
            return None
        indice = faiss.read_index(str(ARCHIVO_INDICE))
    except (OSError, ValueError, RuntimeError):
        return None
    if indice.d != DIMENSION_EMBEDDINGS or indice.ntotal != len(esperado["ids"]):
        return None
    return indice


def construir_o_cargar_indice(documentos: list[dict]) -> faiss.Index:
    esperado = datos_manifiesto(documentos)
    indice = cargar_indice_vigente(esperado)
    if indice is not None:
        print(f"Índice recargado desde {ARCHIVO_INDICE.name}: {indice.ntotal} documentos.")
        print("No se regeneraron embeddings del catálogo.")
        return indice

    print(f"Generando embeddings de {len(documentos)} documentos con {MODELO_EMBEDDINGS}...")
    cliente = crear_cliente()
    textos = [item["descripcion_semantica"] for item in documentos]
    vectores = obtener_embeddings(cliente, textos)
    indice = faiss.IndexFlatL2(DIMENSION_EMBEDDINGS)
    indice.add(vectores)

    DIRECTORIO_INDICES.mkdir(parents=True, exist_ok=True)
    indice_temporal = ARCHIVO_INDICE.with_name(ARCHIVO_INDICE.name + ".tmp")
    manifiesto_temporal = ARCHIVO_MANIFIESTO.with_name(ARCHIVO_MANIFIESTO.name + ".tmp")
    faiss.write_index(indice, str(indice_temporal))
    manifiesto_temporal.write_text(
        json.dumps(esperado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(indice_temporal, ARCHIVO_INDICE)
    os.replace(manifiesto_temporal, ARCHIVO_MANIFIESTO)
    print(f"Índice guardado en {ARCHIVO_INDICE.name}: {indice.ntotal} documentos.")
    return indice


def buscar_consultas(indice: faiss.Index, documentos: list[dict]) -> None:
    cliente = crear_cliente()
    for consulta in CONSULTAS_PRUEBA:
        vector = obtener_embeddings(cliente, [consulta])
        distancias, posiciones = indice.search(vector, min(3, indice.ntotal))
        print(f"\nConsulta: {consulta}")
        for puesto, (distancia, posicion) in enumerate(
            zip(distancias[0], posiciones[0]), start=1
        ):
            documento = documentos[int(posicion)]
            print(
                f"  {puesto}. {documento['id']} | distancia L2²: {float(distancia):.4f} | "
                f"{documento['descripcion_semantica'][:105]}..."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solo-indice",
        action="store_true",
        help="Construye o recarga el índice sin ejecutar las consultas de prueba.",
    )
    opciones = parser.parse_args()
    documentos = cargar_documentos()
    indice = construir_o_cargar_indice(documentos)
    if not opciones.solo_indice:
        buscar_consultas(indice, documentos)


if __name__ == "__main__":
    main()
