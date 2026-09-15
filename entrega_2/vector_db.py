import json
import chromadb


# --- B.1 CARGA DE DATOS A CHROMADB ---

# Leer el archivo JSON que armaron en la Parte A
with open("base_conocimiento.json", "r", encoding="utf-8") as file:
    datos_crudos = json.load(file)

# Crear el cliente persistente de ChromaDB 
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Crear o recuperar la colección
coleccion = chroma_client.get_or_create_collection(
    name="portalia_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Preparar las listas para inyectar en ChromaDB
ids = []
documentos = []
metadatos = []

for item in datos_crudos:
    ids.append(item["id"])
    documentos.append(item["contenido"]) 
    meta = item["metadata"]
    meta["titulo"] = item["titulo"]
    metadatos.append(meta)

# Ingesta de los datos usando UPSERT
coleccion.upsert(
    ids=ids,
    documents=documentos,
    metadatas=metadatos
)

print(f"¡Éxito! Se cargaron {coleccion.count()} documentos en ChromaDB.")



# --- B.3 EVENTO DE NEGOCIO EN CALIENTE ---

print("\n--- Evento en caliente ---")
# Supongamos que Portalia cambia su política y ahora da 45 días para cambios (el ID original era POL-CAM-001)
coleccion.upsert(
    ids=["POL-CAM-001"],
    documents=["Los productos adquiridos en Portalia pueden solicitar un cambio dentro de los 45 días corridos posteriores a la recepción del pedido."],
    metadatas=[{"categoria": "cambios", "activo": True, "tags_regionales": ["AR"]}]
)

# Verificamos que se haya actualizado correctamente
documento_actualizado = coleccion.get(ids=["POL-CAM-001"])
print("Documento actualizado:", documento_actualizado["documents"])



# --- B.4 BÚSQUEDA HÍBRIDA ---

def buscar_portalia(query_semantica: str, categoria: str = None, solo_activos: bool = True, n_resultados: int = 3):
    """
    Realiza una búsqueda híbrida: similitud semántica + filtro duro por metadata.
    """
    # Armamos las condiciones del filtro para pasárselo a ChromaDB
    filtros = []
    
    if categoria:
        filtros.append({"categoria": {"$eq": categoria}})
        
    if solo_activos is not None:
        filtros.append({"activo": {"$eq": solo_activos}})
        
    # Si hay más de un filtro, usamos el operador "$and"
    where_clause = None
    if len(filtros) == 1:
        where_clause = filtros[0]
    elif len(filtros) > 1:
        where_clause = {"$and": filtros}
        
    print(f"\n--- BÚSQUEDA HÍBRIDA ---")
    print(f"Consulta: '{query_semantica}'")
    print(f"Filtro nativo enviado a ChromaDB: {where_clause}")
    
    # Ejecutar la query en ChromaDB
    resultados = coleccion.query(
        query_texts=[query_semantica],
        n_results=n_resultados,
        where=where_clause
    )
    
    # Imprimir los resultados
    for i in range(len(resultados["ids"][0])):
        id_doc = resultados["ids"][0][i]
        distancia = resultados["distances"][0][i]
        documento = resultados["documents"][0][i]
        metadata = resultados["metadatas"][0][i]
        
        print(f"\n[{i+1}] ID: {id_doc} (Distancia: {distancia:.4f})")
        print(f"Metadata: {metadata}")
        print(f"Texto: {documento}")

# --- Prueba de la función ---
buscar_portalia(
    query_semantica="Me vino fallada la remera y la quiero devolver",
    categoria="cambios", 
    solo_activos=True,
    n_resultados=2
)
