"""B.1-B.4: migra la base limpia a ChromaDB y expone la busqueda hibrida de Portalia."""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from embeddings_gemini import (
    DIMENSION_EMBEDDINGS,
    MODELO_EMBEDDINGS,
    crear_cliente,
    obtener_embeddings,
)

DIRECTORIO_ENTREGA = Path(__file__).resolve().parent
ARCHIVO_DATOS = DIRECTORIO_ENTREGA / "base_conocimiento_limpia.json"
DIRECTORIO_CHROMA = DIRECTORIO_ENTREGA / "chroma_db"
NOMBRE_COLECCION = "portalia_knowledge"

# C.2: por encima de esta distancia coseno el resultado deja de considerarse
# "lo bastante parecido" y el sistema responde que no tiene la informacion.
# Calibrado con consultas reales: las consultas del dominio (politicas, cambios,
# devoluciones) devuelven su mejor match entre 0.21 y 0.33 de distancia; las
# consultas fuera de catalogo (futbol, recetas, inmuebles) no bajan de 0.48.
# El umbral se fija en el punto medio de ese rango.
UMBRAL_ACEPTACION = 0.40

SIN_RESULTADOS = (
    "No tengo esa informacion en la base de conocimiento de Portalia."
)


# ============================================================
# B.1 - EMBEDDINGS Y COLECCION
# ============================================================

class EmbeddingsGeminiChroma(EmbeddingFunction):
    """Hace que ChromaDB vectorice con el mismo modelo que el indice FAISS de A.4."""

    def __init__(self) -> None:
        self._cliente = crear_cliente()

    def __call__(self, input: Documents) -> Embeddings:
        return obtener_embeddings(self._cliente, list(input)).tolist()

    @staticmethod
    def name() -> str:
        return "portalia_gemini"

    def get_config(self) -> dict:
        return {"modelo": MODELO_EMBEDDINGS, "dimension": DIMENSION_EMBEDDINGS}

    @staticmethod
    def build_from_config(config: dict) -> "EmbeddingsGeminiChroma":
        return EmbeddingsGeminiChroma()


def abrir_coleccion():
    """B.1: PersistentClient en disco + coleccion con distancia coseno."""
    cliente = chromadb.PersistentClient(path=str(DIRECTORIO_CHROMA))

    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=EmbeddingsGeminiChroma(),
    )


def aplanar_metadatos(metadatos: dict) -> dict:
    """ChromaDB solo admite str, int, float o bool en metadata.

    'tags_regionales' es una lista en el JSON (asi lo pide A.3), por eso se serializa
    a texto separado por comas al entrar a la coleccion.
    """
    plano = {}

    for clave, valor in metadatos.items():
        if isinstance(valor, list):
            plano[clave] = ", ".join(str(item) for item in valor)
        else:
            plano[clave] = valor

    return plano


def cargar_base(coleccion) -> None:
    """B.1: ingesta con upsert para poder re-ejecutar sin duplicar."""
    documentos = json.loads(ARCHIVO_DATOS.read_text(encoding="utf-8"))

    coleccion.upsert(
        ids=[doc["id"] for doc in documentos],
        documents=[doc["descripcion_semantica"] for doc in documentos],
        metadatas=[aplanar_metadatos(doc["metadatos"]) for doc in documentos],
    )

    print(f"Se cargaron {coleccion.count()} documentos en ChromaDB.")


# ============================================================
# B.3 - EVENTO DE NEGOCIO EN CALIENTE
# ============================================================

def evento_en_caliente(coleccion) -> None:
    """B.3: Portalia extiende el plazo de cambios de 30 a 45 dias."""
    antes = coleccion.get(ids=["POL-CAM-001"])
    print("\n--- B.3 EVENTO EN CALIENTE ---")
    print(f"Antes  : {antes['documents'][0]}")
    print(f"Metadata antes: {antes['metadatas'][0]}")

    metadatos = dict(antes["metadatas"][0])

    coleccion.upsert(
        ids=["POL-CAM-001"],
        documents=[
            "Plazo general para cambios. Los productos adquiridos en Portalia pueden "
            "solicitar un cambio dentro de los 45 dias corridos posteriores a la "
            "recepcion del pedido. Para iniciar la solicitud, el cliente debe "
            "identificar el pedido y el producto que desea cambiar."
        ],
        metadatas=[metadatos],
    )

    despues = coleccion.get(ids=["POL-CAM-001"])
    print(f"Despues: {despues['documents'][0]}")
    print(f"Documentos en la coleccion: {coleccion.count()} (sin duplicar el ID)")


# ============================================================
# B.4 - BUSQUEDA HIBRIDA
# ============================================================

def buscar_portalia(
    coleccion,
    query_semantica: str,
    filtro_categoria: str | None = None,
    solo_activos: bool = True,
    solo_con_stock: bool = False,
    n_resultados: int = 3,
    umbral: float = UMBRAL_ACEPTACION,
) -> dict:
    """B.4: semantica + filtro duro nativo. El filtro viaja en el where, nunca en un if."""
    filtros = []

    if filtro_categoria:
        filtros.append({"categoria": {"$eq": filtro_categoria}})

    # solo_activos=False significa "no filtres por vigencia", no "traeme los inactivos".
    if solo_activos:
        filtros.append({"activo": {"$eq": True}})

    if solo_con_stock:
        filtros.append({"en_stock_demo": {"$eq": True}})

    where = None
    if len(filtros) == 1:
        where = filtros[0]
    elif len(filtros) > 1:
        where = {"$and": filtros}

    print("\n--- BUSQUEDA HIBRIDA ---")
    print(f"Consulta: '{query_semantica}'")
    print(f"Filtro nativo enviado a ChromaDB: {where}")

    crudo = coleccion.query(
        query_texts=[query_semantica],
        n_results=n_resultados,
        where=where,
    )

    encontrados = [
        {
            "id": crudo["ids"][0][i],
            "distancia": crudo["distances"][0][i],
            "texto": crudo["documents"][0][i],
            "metadatos": crudo["metadatas"][0][i],
        }
        for i in range(len(crudo["ids"][0]))
    ]

    # C.2: el umbral se aplica sobre la relevancia, no sobre los metadatos.
    # El filtrado por metadatos ya lo hizo ChromaDB en el where.
    aceptados = [item for item in encontrados if item["distancia"] <= umbral]

    if not aceptados:
        peor = f" (mejor distancia: {encontrados[0]['distancia']:.4f})" if encontrados else ""
        print(f"{SIN_RESULTADOS}{peor}")
        return {"respuesta": SIN_RESULTADOS, "documentos": []}

    for posicion, item in enumerate(aceptados, start=1):
        print(f"\n[{posicion}] ID: {item['id']} (distancia: {item['distancia']:.4f})")
        print(f"Metadata: {item['metadatos']}")
        print(f"Texto: {item['texto']}")

    return {"respuesta": None, "documentos": aceptados}


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consulta", nargs="?", help="Texto de la busqueda semantica.")
    parser.add_argument("--categoria", help="Filtro duro por categoria (where nativo).")
    parser.add_argument(
        "--incluir-inactivos",
        action="store_true",
        help="No filtra por vigencia; por defecto solo trae documentos activos.",
    )
    parser.add_argument(
        "--solo-con-stock",
        action="store_true",
        help="Filtra por en_stock_demo=True (where nativo).",
    )
    parser.add_argument("-n", "--n-resultados", type=int, default=3)
    parser.add_argument("--umbral", type=float, default=UMBRAL_ACEPTACION)
    parser.add_argument("--cargar", action="store_true", help="Reingesta la base (B.1).")
    parser.add_argument("--evento-caliente", action="store_true", help="Corre B.3.")
    opciones = parser.parse_args()

    coleccion = abrir_coleccion()

    if opciones.cargar or coleccion.count() == 0:
        cargar_base(coleccion)

    if opciones.evento_caliente:
        evento_en_caliente(coleccion)

    if opciones.consulta:
        buscar_portalia(
            coleccion,
            opciones.consulta,
            filtro_categoria=opciones.categoria,
            solo_activos=not opciones.incluir_inactivos,
            solo_con_stock=opciones.solo_con_stock,
            n_resultados=opciones.n_resultados,
            umbral=opciones.umbral,
        )


if __name__ == "__main__":
    main()
