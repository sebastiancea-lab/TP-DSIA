"""A.5: compara un índice FAISS solo en RAM con otro guardado en disco.

Cada fase debe ejecutarse como un proceso Python independiente. El archivo de
prueba vive junto a este script y nunca modifica el índice principal de A.4.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss

# Al ejecutar este archivo desde a5_prueba_volatilidad/, Python necesita que
# entrega_2/ esté en la ruta de importación para reutilizar el flujo de A.4.
DIRECTORIO_A5 = Path(__file__).resolve().parent
DIRECTORIO_ENTREGA = DIRECTORIO_A5.parent
sys.path.insert(0, str(DIRECTORIO_ENTREGA))

from pipeline_vectorial import (
    DIMENSION_EMBEDDINGS,
    cargar_documentos,
    crear_cliente,
    obtener_embeddings,
)


ARCHIVO_PRUEBA = DIRECTORIO_A5 / "a5_prueba.index"


def construir_indice() -> faiss.Index:
    documentos = cargar_documentos()
    textos = [item["descripcion_semantica"] for item in documentos]
    vectores = obtener_embeddings(crear_cliente(), textos)
    indice = faiss.IndexFlatL2(DIMENSION_EMBEDDINGS)
    indice.add(vectores)
    print(f"Embeddings del catálogo generados por API: {len(textos)}")
    print(f"Vectores en RAM: {indice.ntotal}")
    return indice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fase", choices=("ram", "verificar-ram", "guardar", "recargar")
    )
    fase = parser.parse_args().fase
    print(f"Fase: {fase} | proceso: {os.getpid()} | UTC: {datetime.now(timezone.utc).isoformat()}")

    if fase == "ram":
        if ARCHIVO_PRUEBA.exists():
            raise RuntimeError("Ya existe el índice A.5; no se sobrescribió.")
        indice = construir_indice()
        print("faiss.write_index llamado: no")
        print(f"Archivo de prueba en disco: {ARCHIVO_PRUEBA.exists()}")
        print("Al finalizar este proceso, los vectores de RAM dejan de estar disponibles.")

    elif fase == "verificar-ram":
        if ARCHIVO_PRUEBA.exists():
            raise RuntimeError("El índice A.5 existe; la prueba de RAM no está aislada.")
        print(f"Archivo de prueba en disco: {ARCHIVO_PRUEBA.exists()}")
        print("No hay índice que recargar en este proceso nuevo.")
        print("Para reconstruir los vectores habría que volver a llamar a la API.")

    elif fase == "guardar":
        if ARCHIVO_PRUEBA.exists():
            raise RuntimeError("Ya existe el índice A.5; no se sobrescribió.")
        indice = construir_indice()
        faiss.write_index(indice, str(ARCHIVO_PRUEBA))
        print("faiss.write_index llamado: sí")
        print(f"Archivo de prueba en disco: {ARCHIVO_PRUEBA.exists()}")

    else:
        if not ARCHIVO_PRUEBA.exists():
            raise RuntimeError("Falta el índice A.5 guardado; no se puede recargar.")
        indice = faiss.read_index(str(ARCHIVO_PRUEBA))
        esperados = len(cargar_documentos())
        if indice.ntotal != esperados or indice.d != DIMENSION_EMBEDDINGS:
            raise RuntimeError("El índice recargado no coincide con el catálogo.")
        print(f"Vectores recuperados desde disco: {indice.ntotal}")
        print("Embeddings del catálogo generados por API en este proceso: 0")
        print("faiss.read_index llamado: sí")


if __name__ == "__main__":
    main()
