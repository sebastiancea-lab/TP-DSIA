import json
import unicodedata

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from a2_validacion import similitud_coseno


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_BASE = "base_conocimiento.json"
RUTA_SALIDA = "base_conocimiento_limpia.json"

# Umbral calibrado para documentos de conocimiento.
UMBRAL_DISTANCIA = 0.15
TOP_PARES_DIAGNOSTICO = 15

PARES_CONTROL = [
    ("POL-GAR-001", "POL-GAR-DUP-001"),
    ("ENV-002", "ENV-DUP-001"),
    ("PROC-003", "PROC-DUP-001")
]

# Esquema esperado para los metadatos.
CLAVES_VALIDAS = {
    "categoria",
    "activo",
    "tags_regionales"
}


# ============================================================
# CARGA Y GUARDADO
# ============================================================

def cargar_documentos():
    """Carga la base de conocimiento original."""
    with open(RUTA_BASE, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_documentos(documentos):
    """Guarda el resultado limpio del proceso ETL."""
    with open(RUTA_SALIDA, "w", encoding="utf-8") as archivo:
        json.dump(
            documentos,
            archivo,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# NORMALIZACIÓN ESTRUCTURAL
# ============================================================

def normalizar_texto(texto):
    """
    Normaliza texto para poder comparar claves y valores:
    - elimina espacios al inicio y al final
    - pasa a minúsculas
    - elimina tildes
    """
    texto = texto.strip().lower()

    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def normalizar_claves(metadata, id_documento, cambios):
    """
    Normaliza las claves conocidas de metadata.

    Ejemplo:
    'categoría' -> 'categoria'
    """
    metadata_limpia = {}

    for clave, valor in metadata.items():
        clave_normalizada = normalizar_texto(clave)

        if clave_normalizada in CLAVES_VALIDAS:
            metadata_limpia[clave_normalizada] = valor

            if clave != clave_normalizada:
                cambios.append(
                    f"{id_documento}: clave '{clave}' "
                    f"normalizada a '{clave_normalizada}'"
                )
        else:
            metadata_limpia[clave] = valor

    return metadata_limpia


def normalizar_booleano(valor):
    """
    Convierte distintas representaciones de booleanos
    al tipo bool de Python.
    """
    if isinstance(valor, bool):
        return valor

    if isinstance(valor, str):
        valor_normalizado = normalizar_texto(valor)

        if valor_normalizado in {"true", "1", "si"}:
            return True

        if valor_normalizado in {"false", "0", "no"}:
            return False

    return valor


def normalizar_documento(doc, cambios):
    """Normaliza las claves y tipos de metadata de un documento."""
    metadata = doc.get("metadatos", {})
    id_documento = doc.get("id", "SIN-ID")

    # Normalización de claves.
    metadata = normalizar_claves(
        metadata,
        id_documento,
        cambios
    )

    # Normalización del campo booleano "activo".
    if "activo" in metadata:
        valor_original = metadata["activo"]
        valor_normalizado = normalizar_booleano(valor_original)

        if type(valor_original) != type(valor_normalizado):
            cambios.append(
                f"{id_documento}: campo 'activo' convertido "
                f"de {type(valor_original).__name__} "
                f"a {type(valor_normalizado).__name__}"
            )

        metadata["activo"] = valor_normalizado

    doc["metadatos"] = metadata

    return doc


# ============================================================
# RESOLUCIÓN DE COLISIONES DE IDs
# ============================================================

def resolver_colisiones_ids(documentos, cambios):
    """
    Garantiza que cada documento tenga un ID único.

    Si un ID ya existe:
    DOC-001 -> DOC-001-2 -> DOC-001-3 ...
    """
    ids_usados = set()

    for doc in documentos:
        id_original = doc["id"]
        id_nuevo = id_original
        contador = 2

        while id_nuevo in ids_usados:
            id_nuevo = f"{id_original}-{contador}"
            contador += 1

        if id_nuevo != id_original:
            cambios.append(
                f"Colisión de ID: '{id_original}' "
                f"renombrado a '{id_nuevo}'"
            )

            doc["id"] = id_nuevo

        ids_usados.add(id_nuevo)

    return documentos


# ============================================================
# EMBEDDINGS Y DISTANCIA COSENO
# ============================================================

def generar_embeddings(documentos):
    """Genera un embedding para el contenido de cada documento."""
    modelo = DefaultEmbeddingFunction()

    textos = [
        doc["descripcion_semantica"]
        for doc in documentos
    ]

    return modelo(textos)


def calcular_distancia_coseno(vector_a, vector_b):
    """
    Calcula distancia coseno reutilizando la función
    similitud_coseno implementada en A2.

    Distancia = 1 - similitud
    """
    similitud = similitud_coseno(vector_a, vector_b)

    return 1 - similitud


def encontrar_pares_similares(documentos, embeddings):
    """
    Compara todos los documentos entre sí una única vez
    y ordena los pares desde la menor distancia coseno.
    """
    pares = []

    for i in range(len(documentos)):
        for j in range(i + 1, len(documentos)):
            distancia = calcular_distancia_coseno(
                embeddings[i],
                embeddings[j]
            )

            pares.append({
                "indice_a": i,
                "indice_b": j,
                "id_a": documentos[i]["id"],
                "id_b": documentos[j]["id"],
                "distancia": distancia
            })

    pares.sort(key=lambda par: par["distancia"])

    return pares


def buscar_distancia(par_id_a, par_id_b, pares):
    """Busca la distancia de un par sin importar el orden."""
    ids_buscados = {par_id_a, par_id_b}

    for par in pares:
        if {par["id_a"], par["id_b"]} == ids_buscados:
            return par["distancia"]

    return None


def mostrar_diagnostico_distancias(pares):
    """Muestra los pares más cercanos y las distancias de control."""
    print(
        f"\nTop {TOP_PARES_DIAGNOSTICO} pares con menor distancia coseno:"
    )

    for posicion, par in enumerate(
        pares[:TOP_PARES_DIAGNOSTICO],
        start=1
    ):
        print(
            f"{posicion:02d}. {par['id_a']} <-> {par['id_b']} "
            f"(distancia: {par['distancia']:.4f})"
        )

    print("\nDistancias de pares de control B5:")

    for id_a, id_b in PARES_CONTROL:
        distancia = buscar_distancia(id_a, id_b, pares)

        if distancia is None:
            print(f"- {id_a} <-> {id_b}: no encontrado")
        else:
            print(f"- {id_a} <-> {id_b}: {distancia:.4f}")


def es_producto_catalogo(id_documento):
    """Indica si un documento corresponde al catálogo de productos."""
    return id_documento.startswith("PROD-")


# ============================================================
# PURGA SEMÁNTICA
# ============================================================

def purgar_duplicados_semanticos(documentos, pares, umbral):
    """
    Elimina documentos cuya distancia coseno se encuentre
    por debajo del umbral definido.

    Ante un casi-duplicado, se conserva el documento que
    apareció primero en el dataset y se elimina el posterior.
    """
    indices_a_eliminar = set()
    eliminados = []

    if umbral is None:
        return documentos, eliminados

    for par in pares:
        if (
            es_producto_catalogo(par["id_a"])
            or es_producto_catalogo(par["id_b"])
        ):
            continue

        if par["distancia"] < umbral:
            indice_duplicado = par["indice_b"]

            if indice_duplicado not in indices_a_eliminar:
                indices_a_eliminar.add(indice_duplicado)

                eliminados.append({
                    "id_conservado": par["id_a"],
                    "id_eliminado": par["id_b"],
                    "distancia": par["distancia"]
                })

    documentos_limpios = [
        doc
        for indice, doc in enumerate(documentos)
        if indice not in indices_a_eliminar
    ]

    return documentos_limpios, eliminados


# ============================================================
# EJECUCIÓN DEL ETL
# ============================================================

def main():
    # 1. Cargar dataset original.
    documentos = cargar_documentos()
    cambios = []

    # 2. Normalizar claves y tipos.
    documentos_normalizados = [
        normalizar_documento(doc, cambios)
        for doc in documentos
    ]

    # 3. Resolver posibles colisiones de IDs.
    documentos_normalizados = resolver_colisiones_ids(
        documentos_normalizados,
        cambios
    )

    # 4. Generar embeddings.
    embeddings = generar_embeddings(documentos_normalizados)

    # 5. Calcular distancias entre todos los pares.
    pares_similares = encontrar_pares_similares(
        documentos_normalizados,
        embeddings
    )

    # 6. Realizar la purga semántica.
    documentos_limpios, eliminados = purgar_duplicados_semanticos(
        documentos_normalizados,
        pares_similares,
        UMBRAL_DISTANCIA
    )

    # 7. Guardar la base limpia.
    guardar_documentos(documentos_limpios)

    # ========================================================
    # REPORTE DEL ETL
    # ========================================================

    print("\n=== REPORTE ETL ===")

    print(f"\nDocumentos cargados: {len(documentos)}")
    print(f"Documentos normalizados: {len(documentos_normalizados)}")
    print(f"Embeddings generados: {len(embeddings)}")
    print(f"Dimensiones por embedding: {len(embeddings[0])}")

    print("\nCorrecciones estructurales:")

    if cambios:
        for cambio in cambios:
            print(f"- {cambio}")
    else:
        print("- No se encontraron inconsistencias estructurales.")

    mostrar_diagnostico_distancias(pares_similares)

    print("\nPurga semántica:")

    if eliminados:
        for eliminado in eliminados:
            print(
                f"- Se elimina {eliminado['id_eliminado']} "
                f"por similitud con {eliminado['id_conservado']} "
                f"(distancia: {eliminado['distancia']:.4f})"
            )
    else:
        print("- No se detectaron casi-duplicados.")

    print(
        f"\nDocumentos guardados en la salida: "
        f"{len(documentos_limpios)}"
    )

    print(
        f"Base limpia guardada en: "
        f"{RUTA_SALIDA}"
    )


if __name__ == "__main__":
    main()
