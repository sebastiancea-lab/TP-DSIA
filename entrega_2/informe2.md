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
La base de conocimiento útil de Portalia consta de 38 documentos: 18 sobre políticas de cambios, devoluciones y garantías, envíos, preguntas frecuentes y procedimientos; y 20 descripciones de productos de electrónica. Cada documento tiene un `id` único, un párrafo en `descripcion_semantica` y un objeto `metadatos`. Las descripciones de productos incluyen características y contextos de uso; el catálogo contiene variantes similares —como celulares 4G y 5G o televisores LED y QLED— para poner a prueba la recuperación.

Todos los documentos comparten tres metadatos: `categoria`, que identifica el tipo de contenido; `activo`, que indica su vigencia; y `tags_regionales`, una lista de sinónimos y expresiones comerciales relacionadas. Actualmente los 38 documentos están activos. Los productos agregan `en_stock_demo` para ensayar filtros de disponibilidad: 15 tienen el valor `true` y 5, `false`. Según la categoría, incorporan además especificaciones filtrables, como `red_movil`, `almacenamiento_gb`, `tecnologia_panel`, `ram_gb`, `cancelacion_activa`, `usb_c_video_carga` o `potencia_w`. Un campo específico se omite cuando no corresponde al documento, en lugar de completarlo con un valor ficticio.

Aplicamos la Regla del Arquitecto separando lo narrativo de los datos que necesitan filtros estrictos. El significado y el contexto de uso quedan en `descripcion_semantica`, mientras que la categoría, la vigencia, la disponibilidad simulada y las especificaciones comprobables quedan en `metadatos`. Los valores de `tags_regionales` son etiquetas de apoyo: no modifican por sí solos el texto que se vectoriza. `en_stock_demo` permite probar búsquedas con y sin disponibilidad; en el funcionamiento previsto para Portalia, ese estado se actualizaría desde el sistema de inventario.

Para la prueba de ETL de B.5, posteriormente se añadieron al archivo de entrada `base_conocimiento.json` tres documentos casi duplicados y dos inconsistencias estructurales intencionales. Por eso ese archivo contiene ahora 41 registros; `base_conocimiento_limpia.json` conserva los 38 documentos útiles después de normalizar y eliminar los tres duplicados. Las pruebas de A.4 y A.5 que se describen a continuación se realizaron antes de esa ampliación, con los 38 documentos originales.

## A.4 — Generación de embeddings e índice FAISS
El script `pipeline_vectorial.py` toma la `descripcion_semantica` de cada documento de `base_conocimiento.json`, genera embeddings con el modelo configurado en `.env` y los incorpora a un índice FAISS `IndexFlatL2`. El índice se guarda en `indices/portalia.index` junto con un manifiesto que registra el modelo, la dimensión, los identificadores y la huella del archivo de entrada. Si el archivo o la configuración cambian, el índice anterior deja de considerarse vigente y se reconstruye.

La implementación se probó con los 38 documentos previos a B.5 y consultas sobre productos y políticas. El archivo de entrada actual contiene 41 registros por los casos intencionales de B.5; una nueva ejecución de A.4 generaría un índice de 41 vectores y no corresponde a la prueba anterior de 38.

## A.5 — Prueba de volatilidad de la RAM
La prueba se realizó antes de la ampliación del archivo para B.5, con los 38 documentos originales y cuatro procesos Python independientes: se construyó un índice de 38 vectores sin guardarlo y el proceso siguiente no encontró ningún archivo para recargar. Luego se generaron nuevamente los embeddings, se guardó el índice con `faiss.write_index()` y otro proceso recuperó los 38 vectores con `faiss.read_index()` sin regenerarlos. Los comandos, horarios y salidas se encuentran en `a5_prueba_volatilidad/evidencia_a5.md`.

Si el servidor se reinicia y el índice solo estaba en RAM, debe regenerar los embeddings; con `write_index` puede recargar los vectores guardados sin volver a generarlos.
Con dos servidores, cada uno necesita acceso a la misma versión del índice y del catálogo, o un mecanismo coordinado de actualización, para evitar respuestas basadas en estados diferentes.


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


## B.5 — ETL de normalización y purga semántica

Para probar el proceso ETL se preparó intencionalmente el dataset incorporando tres casi-duplicados semánticos: `POL-GAR-DUP-001`, similar a `POL-GAR-001`; `ENV-DUP-001`, similar a `ENV-002`; y `PROC-DUP-001`, similar a `PROC-003`. Además, se agregaron dos inconsistencias estructurales: `FAQ-003` contiene inicialmente la clave `"categoría"` en lugar de `"categoria"`, y `PROC-002` contiene inicialmente `"activo": "true"` como string en lugar de booleano.

El script `etl_purga.py` normaliza nombres de claves, normaliza tipos booleanos y resuelve de forma genérica posibles colisiones de IDs mediante sufijos incrementales, garantizando identificadores únicos sin asumir casos particulares. Luego genera embeddings con `DefaultEmbeddingFunction` de ChromaDB y reutiliza la función `similitud_coseno` implementada en A2. La distancia utilizada se calcula como:

`distancia_coseno = 1 - similitud_coseno`

Para la purga semántica se utilizó `UMBRAL_DISTANCIA = 0.15`, calibrado para este dataset y este modelo de embeddings. No debe interpretarse como un valor universal. Las distancias de los casi-duplicados fueron:

- `ENV-002` <-> `ENV-DUP-001`: 0.0103
- `PROC-003` <-> `PROC-DUP-001`: 0.0232
- `POL-GAR-001` <-> `POL-GAR-DUP-001`: 0.0304

Los tres pares quedaron por debajo del umbral y se eliminó el documento posterior en cada caso. Durante la calibración también se detectó que algunos productos legítimos del catálogo tienen distancias coseno muy bajas por ser variantes comerciales con descripciones similares, por ejemplo:

- `PROD-007` <-> `PROD-008`: 0.0337
- `PROD-009` <-> `PROD-010`: 0.0350
- `PROD-001` <-> `PROD-002`: 0.0716

Estos registros no son duplicados. Por ese motivo, la purga semántica excluye los documentos cuyo ID comienza con `"PROD-"`, ya que una similitud semántica alta entre dos productos no implica que representen el mismo producto. Entre los documentos no pertenecientes al catálogo, no se detectaron otros pares legítimos por debajo del umbral 0.15.

El resultado del ETL fue:

- Documentos de entrada: 41
- Documentos finales: 38
- Documentos eliminados: `ENV-DUP-001`, `PROC-DUP-001` y `POL-GAR-DUP-001`
- Correcciones estructurales: `FAQ-003`, `"categoría"` -> `"categoria"`; `PROC-002`, `"activo"` de string a booleano

Un `SELECT DISTINCT` no habría detectado estos casos, porque solo identifica registros con valores exactamente iguales. En este dataset los documentos tienen IDs y textos diferentes, aunque expresan prácticamente el mismo concepto. La detección mediante embeddings y distancia coseno permite identificar similitud semántica que no es visible mediante igualdad textual exacta.

