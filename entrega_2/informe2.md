# TP Integrador — Entrega 2
## Del Prompt Saturado a la Base de Conocimiento Vectorial

## Parte A — Embeddings y Búsqueda Semántica
### A.1 — Autopsia del contexto estático
Nosotros definimos que Portalia incluiría políticas de cambios, devoluciones y garantías, preguntas frecuentes, procedimientos de postventa e información descriptiva del catálogo.

Si toda esta información se incluyera directamente dentro del System Prompt en cada consulta realizada por un cliente, aparecerían principalmente tres problemas:

| Problema | Aplicado a Portalia |
|---|---|
| **Desangre de tokens** | Para esta entrega, la Base de Conocimiento inicial de Portalia tendrá al menos 15 documentos, aunque en un sistema real podría crecer a cientos o miles de políticas, procedimientos, preguntas frecuentes y descripciones de productos. Si todos esos documentos se enviaran dentro del prompt en cada consulta, el modelo tendría que procesarlos nuevamente aunque el cliente necesitara solamente una pequeña parte de esa información. Esto aumentaría innecesariamente la cantidad de tokens procesados, el costo de uso de la API y el tiempo de respuesta. |
| **Lost in the Middle** | Al incluir una gran cantidad de información en un mismo contexto, el dato relevante puede quedar perdido entre muchos documentos que no tienen relación con la consulta. Por ejemplo, si un cliente pregunta *“¿cuántos días tengo para cambiar una remera?”*, la política específica sobre el plazo de cambios podría quedar en medio de información sobre garantías, devoluciones, envíos y otros procedimientos. Esto aumenta el riesgo de que el modelo no priorice correctamente el fragmento necesario y genere una respuesta incompleta o incorrecta. |
| **Inconsistencia de estado concurrente** | Algunos datos de Portalia cambian constantemente. Durante una misma sesión, el stock de una variante puede agotarse, un pedido puede pasar de *“en preparación”* a *“despachado”* o una solicitud de devolución puede cambiar de estado. Si estos datos estuvieran copiados dentro del prompt, podrían quedar desactualizados respecto de la base de datos o de la API de logística. Por eso, como se definió en la Entrega 1, los datos dinámicos deben consultarse en tiempo real desde sus fuentes de verdad y no formar parte del contexto estático del LLM. |

#### ¿Por qué `SELECT ... WHERE descripcion LIKE '%...%'` tampoco alcanza?
Una búsqueda mediante `LIKE` se basa en coincidencias textuales y no comprende el significado de una consulta. Por ejemplo, un cliente podría escribir *“la remera me queda chica y quiero otro talle”*, mientras que el documento de Portalia utiliza términos como *“cambio de producto”*. Aunque ambos textos se refieran al mismo concepto, una búsqueda basada únicamente en palabras podría no relacionarlos. La búsqueda semántica permite recuperar información por similitud de significado, aunque el cliente utilice palabras, expresiones o formas de escribir diferentes de las presentes en el documento.

## A.2 — Similitud coseno a mano

Para representar de manera simplificada el dominio de Portalia definimos dos ejes:

- **Eje X:** relación con cambios y devoluciones.
- **Eje Y:** relación con envíos y pedidos.

A partir de estos ejes representamos tres documentos y una consulta mediante vectores 2D:

- **Documento A = [9, 2]:** Política de cambios y devoluciones.
- **Documento B = [2, 9]:** Seguimiento de pedidos y envíos.
- **Documento C = [7, 5]:** Problemas postventa generales.
- **Consulta Q = [8, 3]:** "Quiero cambiar una remera que me quedó chica".

La similitud coseno se calcula mediante:

Similitud(A, B) = (A · B) / (||A|| × ||B||)

### Consulta Q vs. Documento A

**1. Producto punto**

Q · A = (8 × 9) + (3 × 2)  
Q · A = 72 + 6 = 78

**2. Norma de los vectores**

||Q|| = √(8² + 3²) = √73 ≈ 8.54

||A|| = √(9² + 2²) = √85 ≈ 9.22

**3. Similitud coseno**

sim(Q,A) = 78 / (8.54 × 9.22)

sim(Q,A) ≈ 0.99

---

### Consulta Q vs. Documento B

**1. Producto punto**

Q · B = (8 × 2) + (3 × 9)  
Q · B = 16 + 27 = 43

**2. Norma de los vectores**

||Q|| = √73 ≈ 8.54

||B|| = √(2² + 9²) = √85 ≈ 9.22

**3. Similitud coseno**

sim(Q,B) = 43 / (8.54 × 9.22)

sim(Q,B) ≈ 0.55

---

### Consulta Q vs. Documento C

**1. Producto punto**

Q · C = (8 × 7) + (3 × 5)  
Q · C = 56 + 15 = 71

**2. Norma de los vectores**

||Q|| = √73 ≈ 8.54

||C|| = √(7² + 5²) = √74 ≈ 8.60

**3. Similitud coseno**

sim(Q,C) = 71 / (8.54 × 8.60)

sim(Q,C) ≈ 0.97

### Resultados

Los resultados obtenidos son:

- Documento A: ≈ 0.99
- Documento C: ≈ 0.97
- Documento B: ≈ 0.55

El Documento A presenta la mayor similitud con la consulta, lo cual resulta coherente ya que ambos están fuertemente relacionados con cambios y devoluciones. El Documento C también presenta una similitud alta por tratar problemas generales de postventa, mientras que el Documento B está principalmente relacionado con pedidos y envíos.


## A.3 — Construcción de `base_conocimiento.json`
### Diseño del esquema — Regla del Arquitecto
La Base de Conocimiento se organizó en un único archivo `base_conocimiento.json` que contiene 18 registros de conocimiento independientes. Cada registro es considerado un documento y lo procesaremos individualmente durante la generación de embeddings.

Para cada documento se definieron los siguientes campos:
- `id`: identificador único que permite reconocer cada documento.
- `titulo`: nombre que resume el tema principal del documento.
- `descripcion_semantica`: párrafo que contiene la información semánticamente relevante que despues será transformada en un embedding. Se utilizan descripciones completas y contextualizadas en lugar de palabras aisladas para mejorar la recuperación por similitud semántica.
- `metadata.categoria`: campo categórico que permite clasificar los documentos según su función dentro de Portalia, por ejemplo cambios, devoluciones, garantías o envíos.
- `metadata.activo`: campo booleano que permite distinguir documentos vigentes de documentos que ya no deberían utilizarse.
- `metadata.tags_regionales`: permite identificar las regiones en las que resulta aplicable cada documento.

Siguiendo la Regla del Arquitecto, se separó la información que debe participar en la búsqueda semántica de aquella que resulta más adecuada para realizar filtros estructurados. La `descripcion_semantica` contiene el significado que queremos comparar mediante embeddings, mientras que campos como `categoria`, `activo` y `tags_regionales` se almacenan como metadata para poder aplicar filtros durante la recuperación.



## Parte B — ChromaDB, Filtrado Híbrido y ETL

### B.1 — Migración a ChromaDB
La migración de la base de conocimiento estática (`base_conocimiento.json`) hacia ChromaDB fue implementada en el script `vector_db.py`. Para esto se utilizó un `PersistentClient` que guarda la información en disco dentro del directorio `chroma_db/`, configurando la colección con distancia `cosine` e inyectando los datos mediante la operación `upsert` para evitar duplicados en ejecuciones posteriores.


## B.2 — Los tres límites de FAISS que ChromaDB resuelve

| Límite de FAISS | Cómo se manifiesta en su dominio | Cómo lo resuelve ChromaDB |
|---|---|---|
| **Sin persistencia transaccional / atomicidad** | En FAISS el índice vive en la memoria RAM y solo se guarda si llamamos manualmente a `write_index()`. Si el servidor de Portalia se reinicia antes de guardar, se pierden todos los nuevos documentos ingresados. | ChromaDB utiliza una base de datos embebida (SQLite) y guarda los datos en disco de forma transaccional. Cada operación se persiste automáticamente, resolviendo la volatilidad de la RAM. |
| **Sin filtrado híbrido nativo** | FAISS solo sabe calcular distancias matemáticas. Si un cliente de Portalia pregunta por "políticas de envío", y queremos filtrar solo documentos vigentes (`activo: true`), FAISS nos obligaría a buscar todo primero y filtrar después (descartando resultados y perdiendo eficiencia). | ChromaDB permite realizar "filtrado híbrido". Acepta condiciones en la consulta (usando el parámetro `where`), filtrando la metadata *antes* de calcular la similitud vectorial. |
| **CRUD ineficiente / sin concurrencia** | FAISS no permite actualizar o eliminar un documento específico fácilmente por su ID. Si actualizamos una política de devoluciones en Portalia, reconstruir el índice de FAISS es costoso. | ChromaDB ofrece operaciones CRUD completas y nativas (`add`, `update`, `upsert`, `delete`) basadas en el ID único de cada documento, permitiendo actualización en caliente. |


## B.3 — Evento de negocio en caliente

Simulamos la actualización de la política de cambios (ID: `POL-CAM-001`), extendiendo el plazo a 45 días utilizando el comando `coleccion.upsert()`. Al verificar con `coleccion.get()`, comprobamos que el texto se actualizó exitosamente sin crear duplicados.

**¿Por qué usamos `upsert` y no `add` ni `update`?**
Usamos `upsert` ("update or insert") porque vuelve la operación idempotente: si el documento no existe, lo crea (como haría `add`); si ya existe, lo actualiza (como haría `update`). Si usáramos `add`, el sistema lanzaría un error al encontrar un ID duplicado, y si usáramos `update` fallaría si el documento aún no fue ingresado.



## B.4 — Búsqueda Híbrida

La función de búsqueda híbrida fue implementada en el script `vector_db.py` (`buscar_portalia`). Para cumplir con las reglas de eficiencia y evitar el post-filtering manual, los filtros duros (por ejemplo, buscar solo documentos de la categoría "cambios" y que estén activos) se resuelven de forma nativa utilizando el operador `$and`. 

De esta manera, el filtro se inyecta directamente en la cláusula `where` de ChromaDB, descartando los documentos irrelevantes antes de que el motor de la base de datos gaste recursos calculando la similitud semántica.

