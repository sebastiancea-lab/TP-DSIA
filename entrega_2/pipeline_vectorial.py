"""A.4: construye o recarga un índice FAISS del catálogo de Portalia."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np

from embeddings_gemini import (
    DIMENSION_EMBEDDINGS,
    MODELO_EMBEDDINGS,
    crear_cliente,
    obtener_embeddings,
)


DIRECTORIO_ENTREGA = Path(__file__).resolve().parent
# Las rutas parten de este archivo para que el script funcione igual desde la
# raiz del repositorio o desde entrega_2.
# A.4 consume la salida limpia del ETL de B.5, no la base cruda con duplicados
# e inconsistencias que se usan para probar ese proceso.
ARCHIVO_DATOS = DIRECTORIO_ENTREGA / "base_conocimiento_limpia.json"
DIRECTORIO_INDICES = DIRECTORIO_ENTREGA / "indices"
ARCHIVO_INDICE = DIRECTORIO_INDICES / "portalia.index"
ARCHIVO_MANIFIESTO = DIRECTORIO_INDICES / "portalia.index.meta.json"


# Consultas representativas para comprobar que la busqueda semantica recupera
# productos y politicas relacionados, aunque no coincidan literalmente.
CONSULTAS_PRUEBA = (
    "Necesito un teléfono con red 5G y espacio para guardar muchos videos.",
    "Busco una pantalla grande con tecnología QLED para ver películas.",
    "Me arrepentí de la compra: ¿hasta cuándo puedo pedir una devolución?",
)


def cargar_documentos() -> list[dict]:
    # Cada vector del indice debe poder asociarse a un documento con ID unico y
    # descripcion para generar su embedding.
    documentos = json.loads(ARCHIVO_DATOS.read_text(encoding="utf-8"))
    if not documentos or any(not item.get("descripcion_semantica") for item in documentos):
        raise ValueError("La base debe contener descripciones semánticas no vacías.")
    ids = [item["id"] for item in documentos]
    if len(ids) != len(set(ids)):
        raise ValueError("La base contiene IDs duplicados.")
    return documentos


def datos_manifiesto(documentos: list[dict]) -> dict:
    # La huella detecta cambios en el catalogo; modelo, dimension e IDs permiten
    # comprobar que el indice corresponde a los documentos y embeddings actuales.
    return {
        "sha256_catalogo": hashlib.sha256(ARCHIVO_DATOS.read_bytes()).hexdigest(),
        "modelo": MODELO_EMBEDDINGS,
        "dimension": DIMENSION_EMBEDDINGS,
        "ids": [item["id"] for item in documentos],
    }


def cargar_indice_vigente(esperado: dict) -> faiss.Index | None:
    # Para reutilizar el indice tienen que existir tanto el archivo FAISS como
    # su manifiesto, y este debe coincidir con la entrada actual.
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
        # Si la fuente no cambio, se recarga desde disco y se evita pagar por
        # generar nuevamente todos los embeddings del catalogo.
        print(f"Índice recargado desde {ARCHIVO_INDICE.name}: {indice.ntotal} documentos.")
        print("No se regeneraron embeddings del catálogo.")
        return indice

    # Si falta el indice o esta desactualizado, vectorizamos cada descripcion
    # semantica usando el modelo configurado para la Entrega 2.
    print(f"Generando embeddings de {len(documentos)} documentos con {MODELO_EMBEDDINGS}...")
    cliente = crear_cliente()
    textos = [item["descripcion_semantica"] for item in documentos]
    vectores = obtener_embeddings(cliente, textos)
    # IndexFlatL2 compara cada vector con todos los demas mediante distancia
    # euclidiana al cuadrado; distancias menores indican mayor cercania.
    indice = faiss.IndexFlatL2(DIMENSION_EMBEDDINGS)
    indice.add(vectores)

    # Escribimos primero archivos temporales y luego los reemplazamos para no
    # dejar un indice o manifiesto parcialmente escrito si la ejecucion falla.
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
    # Las consultas se vectorizan con el mismo modelo del indice; FAISS devuelve
    # las posiciones de los vecinos mas cercanos, que corresponden al orden
    # usado al cargar los documentos.
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
    # --solo-indice sirve para preparar la persistencia sin gastar embeddings
    # adicionales en las consultas de demostracion.
    if not opciones.solo_indice:
        buscar_consultas(indice, documentos)


if __name__ == "__main__":
    main()
