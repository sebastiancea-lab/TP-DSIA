# TP Integrador — Entrega 2
## Del Prompt Saturado a la Base de Conocimiento Vectorial

## Parte A — Embeddings y Búsqueda Semántica
### A.1 — Autopsia del contexto estático
Nosotros definimos que Portalia incluiría políticas de cambios, devoluciones y garantías, preguntas frecuentes, procedimientos de postventa e información descriptiva del catálogo.

Si toda esta información se incluyera directamente dentro del System Prompt en cada consulta realizada por un cliente, aparecerían principalmente tres problemas:

| Problema | Aplicado a Portalia |
|---|---|
| **Desangre de tokens** | La Base de Conocimiento de Portalia tiene 38 documentos. Medido con `tiktoken` (encoding `cl100k_base`, el mismo que usa `parte_a/tiktoken_test.py`), solo las `descripcion_semantica` ocupan 2.808 tokens, y el JSON completo con metadatos ocupa 5.724 tokens. Si esos 5.724 tokens se mandaran dentro del System Prompt en cada consulta, con un volumen modesto de 1.000 consultas diarias el sistema procesaría cerca de 5,7 millones de tokens de entrada por día (~172 millones por mes) solo para que el modelo lea 38 documentos completos cuando la consulta necesita, en general, uno o dos. El problema no es lineal: si el catálogo de Portalia creciera a 500 documentos (un salto realista para un e-commerce en producción), el contexto por consulta rondaría los 75.000 tokens, muy por encima de lo que conviene pagar y procesar en cada intercambio. Con la base vectorial, el costo por consulta pasa a ser el de generar el embedding de la pregunta y recuperar los 3-5 documentos relevantes, sin importar cuánto crezca el catálogo. |
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

### ¿A partir de qué score un resultado es "lo bastante parecido"?

`sim(Q,C) ≈ 0.97` queda muy cerca de `sim(Q,A) ≈ 0.99`: en este ejemplo reducido a 2 ejes, dos documentos con temáticas distintas (cambios/devoluciones vs. postventa general) pueden dar una similitud casi idéntica. Sin un umbral, el sistema respondería siempre con el resultado más cercano, sea o no relevante. Hace falta un corte por debajo del cual el sistema admita que no tiene una coincidencia suficientemente buena.

En la implementación real (`vector_db.py`, ChromaDB) el corte se define sobre **distancia** coseno (`1 - similitud`) y no sobre similitud, porque es lo que devuelve la API de consulta. Si ninguna distancia recuperada queda por debajo del umbral, el sistema responde que no tiene esa información en vez de forzar el resultado más cercano — forzarlo sería alucinación, exactamente el problema que ya se había documentado en la Entrega 1 cuando Gemini inventó datos del pedido 48327 sin tener una fuente real. El valor concreto de ese umbral y cómo se calibró están en C.2.

## A.3 — Construcción de `base_conocimiento.json`
La base de conocimiento útil de Portalia consta de 38 documentos: 18 sobre políticas de cambios, devoluciones y garantías, envíos, preguntas frecuentes y procedimientos; y 20 descripciones de productos de electrónica. Cada documento tiene un `id` único, un párrafo en `descripcion_semantica` y un objeto `metadatos`. Las descripciones de productos incluyen características y contextos de uso; el catálogo contiene variantes similares —como celulares 4G y 5G o televisores LED y QLED— para poner a prueba la recuperación.

Todos los documentos comparten tres metadatos: `categoria`, que identifica el tipo de contenido; `activo`, que indica su vigencia; y `tags_regionales`, una lista de sinónimos y expresiones comerciales relacionadas. Actualmente los 38 documentos están activos. Los productos agregan `en_stock_demo` para ensayar filtros de disponibilidad: 15 tienen el valor `true` y 5, `false`. Según la categoría, incorporan además especificaciones filtrables, como `red_movil`, `almacenamiento_gb`, `tecnologia_panel`, `ram_gb`, `cancelacion_activa`, `usb_c_video_carga` o `potencia_w`. Un campo específico se omite cuando no corresponde al documento, en lugar de completarlo con un valor ficticio.

Aplicamos la Regla del Arquitecto separando lo narrativo de los datos que necesitan filtros estrictos. El significado y el contexto de uso quedan en `descripcion_semantica`, mientras que la categoría, la vigencia, la disponibilidad simulada y las especificaciones comprobables quedan en `metadatos`. Los valores de `tags_regionales` son etiquetas de apoyo: no modifican por sí solos el texto que se vectoriza. `en_stock_demo` permite probar búsquedas con y sin disponibilidad; en el funcionamiento previsto para Portalia, ese estado se actualizaría desde el sistema de inventario.

Para la prueba de ETL de B.5, posteriormente se añadieron al archivo de entrada `base_conocimiento.json` tres documentos casi duplicados y dos inconsistencias estructurales intencionales. Por eso ese archivo contiene ahora 41 registros y es la entrada cruda del ETL, nunca la fuente directa de los índices: `etl_purga.py` lo normaliza y purga, y escribe `base_conocimiento_limpia.json` con los 38 documentos útiles. El índice FAISS de A.4 y la colección de ChromaDB de B.1 se construyen a partir de ese archivo limpio, así que ambos quedan siempre en 38 documentos aunque la base cruda crezca con nuevos casos de prueba.

## A.4 — Generación de embeddings e índice FAISS
El script `pipeline_vectorial.py` toma la `descripcion_semantica` de cada documento de `base_conocimiento_limpia.json` (la salida del ETL de B.5), genera embeddings con el modelo configurado en `.env` y los incorpora a un índice FAISS `IndexFlatL2`. El índice se guarda en `indices/portalia.index` junto con un manifiesto que registra el modelo, la dimensión, los identificadores y la huella del archivo de entrada. Si el archivo o la configuración cambian, el índice anterior deja de considerarse vigente y se reconstruye.

La implementación se probó con los 38 documentos de la base limpia y tres consultas de prueba sobre productos y políticas. Por ejemplo, la consulta "Necesito un teléfono con red 5G y espacio para guardar muchos videos" recupera en primer lugar `PROD-003` (Celular Portalia Nova 5G 256, distancia L2² 0.1931), y "Me arrepentí de la compra: ¿hasta cuándo puedo pedir una devolución?" recupera `POL-DEV-001` (distancia L2² 0.1294) antes que la política de cambios, mostrando que el índice distingue devoluciones de cambios aunque ambas hablen de "arrepentimiento" de compra.

## A.5 — Prueba de volatilidad de la RAM
La prueba se realizó con los 38 documentos de `base_conocimiento_limpia.json` y cuatro procesos Python independientes: se construyó un índice de 38 vectores sin guardarlo y el proceso siguiente no encontró ningún archivo para recargar. Luego se generaron nuevamente los embeddings, se guardó el índice con `faiss.write_index()` y otro proceso recuperó los 38 vectores con `faiss.read_index()` sin regenerarlos. Los comandos, horarios y salidas se encuentran en `a5_prueba_volatilidad/evidencia_a5.md`.

Si el servidor se reinicia y el índice solo estaba en RAM, debe regenerar los embeddings; con `write_index` puede recargar los vectores guardados sin volver a generarlos.
Con dos servidores, cada uno necesita acceso a la misma versión del índice y del catálogo, o un mecanismo coordinado de actualización, para evitar respuestas basadas en estados diferentes.


## Parte B — ChromaDB, Filtrado Híbrido y ETL

### B.1 — Migración a ChromaDB
La migración de la base de conocimiento hacia ChromaDB fue implementada en el script `vector_db.py`, a partir de `base_conocimiento_limpia.json` (la salida del ETL de B.5, no el archivo crudo con los casi-duplicados intencionales). Para esto se utilizó un `PersistentClient` que guarda la información en disco dentro del directorio `chroma_db/`, configurando la colección con distancia `cosine` e inyectando los datos mediante la operación `upsert` para evitar duplicados en ejecuciones posteriores.

La colección vectoriza con el mismo modelo que A.4 y B.5 (`gemini-embedding-001`, 768 dimensiones), mediante una `EmbeddingFunction` propia (`EmbeddingsGeminiChroma`) que reutiliza `embeddings_gemini.py`. `tags_regionales` es una lista en el JSON, pero ChromaDB solo admite `str`, `int`, `float` o `bool` en los metadatos de un documento; por eso `aplanar_metadatos()` la serializa a texto separado por comas antes de la ingesta (por ejemplo, `"cambio, plazo, 30 días"`). Corrida real:

```text
$ uv run python entrega_2/vector_db.py --cargar
Se cargaron 38 documentos en ChromaDB.
```

Volver a correr el mismo comando no duplica documentos: `coleccion.count()` sigue devolviendo 38, porque la ingesta usa `upsert` sobre los mismos IDs.


## B.2 — Los tres límites de FAISS que ChromaDB resuelve

| Límite de FAISS | Cómo se manifiesta en su dominio | Cómo lo resuelve ChromaDB |
|---|---|---|
| **Sin persistencia transaccional / atomicidad** | En FAISS el índice vive en la memoria RAM y solo se guarda si llamamos manualmente a `write_index()`. Si el servidor de Portalia se reinicia antes de guardar, se pierden todos los nuevos documentos ingresados. | ChromaDB utiliza una base de datos embebida (SQLite) y guarda los datos en disco de forma transaccional. Cada operación se persiste automáticamente, resolviendo la volatilidad de la RAM. |
| **Sin filtrado híbrido nativo** | FAISS solo sabe calcular distancias matemáticas. Si un cliente de Portalia pregunta por "políticas de envío", y queremos filtrar solo documentos vigentes (`activo: true`), FAISS nos obligaría a buscar todo primero y filtrar después (descartando resultados y perdiendo eficiencia). | ChromaDB permite realizar "filtrado híbrido". Acepta condiciones en la consulta (usando el parámetro `where`), filtrando la metadata *antes* de calcular la similitud vectorial. |
| **CRUD ineficiente / sin concurrencia** | FAISS no permite actualizar o eliminar un documento específico fácilmente por su ID. Si actualizamos una política de devoluciones en Portalia, reconstruir el índice de FAISS es costoso. | ChromaDB ofrece operaciones CRUD completas y nativas (`add`, `update`, `upsert`, `delete`) basadas en el ID único de cada documento, permitiendo actualización en caliente. |


## B.3 — Evento de negocio en caliente

Simulamos la actualización de la política de cambios (ID: `POL-CAM-001`), extendiendo el plazo a 45 días utilizando el comando `coleccion.upsert()`. Al verificar con `coleccion.get()`, comprobamos que el texto se actualizó exitosamente sin crear duplicados. Corrida real (`uv run python entrega_2/vector_db.py --evento-caliente`):

```text
--- B.3 EVENTO EN CALIENTE ---
Antes  : Plazo general para cambios. [...] dentro de los 30 días corridos [...]
Metadata antes: {'categoria': 'cambios', 'activo': True, 'tags_regionales': 'cambio, plazo, 30 días'}
Después: Plazo general para cambios. [...] dentro de los 45 días corridos [...]
Documentos en la colección: 38 (sin duplicar el ID)
```

El conteo de la colección se mantiene en 38 antes y después del `upsert`: se actualizó el documento existente, no se agregó uno nuevo.

**¿Por qué usamos `upsert` y no `add` ni `update`?**
Usamos `upsert` ("update or insert") porque vuelve la operación idempotente: si el documento no existe, lo crea (como haría `add`); si ya existe, lo actualiza (como haría `update`). Si usáramos `add`, el sistema lanzaría un error al encontrar un ID duplicado, y si usáramos `update` fallaría si el documento aún no fue ingresado.



## B.4 — Búsqueda Híbrida

La función de búsqueda híbrida fue implementada en el script `vector_db.py` (`buscar_portalia`) y expuesta como CLI con `argparse`. Para cumplir con las reglas de eficiencia y evitar el post-filtering manual, los filtros duros (categoría, vigencia y, opcionalmente, disponibilidad simulada) se resuelven de forma nativa utilizando el operador `$and`.

De esta manera, el filtro se inyecta directamente en la cláusula `where` de ChromaDB, descartando los documentos irrelevantes antes de que el motor de la base de datos gaste recursos calculando la similitud semántica: la similitud nunca se calcula sobre un documento que el filtro iba a descartar.

Corrida real (`uv run python entrega_2/vector_db.py "me queda chica la remera y quiero otro talle" --categoria cambios`):

```text
--- BUSQUEDA HIBRIDA ---
Consulta: 'me queda chica la remera y quiero otro talle'
Filtro nativo enviado a ChromaDB: {'$and': [{'categoria': {'$eq': 'cambios'}}, {'activo': {'$eq': True}}]}

[1] ID: POL-CAM-004 (distancia: 0.3392)
[2] ID: POL-CAM-003 (distancia: 0.3674)
[3] ID: POL-CAM-001 (distancia: 0.3797)
```

Los tres resultados pertenecen a la categoría `cambios` y están activos, aunque la consulta no menciona esas palabras: la relevancia se resolvió por semántica ("me queda chica" → cambio de variante/talle) y el alcance se acotó por el `where`.

El umbral de aceptación (`UMBRAL_ACEPTACION`, ver C.2) se aplica **después** del `where`, sobre la lista ya filtrada: no es post-filtering, porque no descarta por metadatos, sino que decide si lo recuperado es lo bastante parecido a la consulta como para responder con eso en vez de admitir que no se tiene la información.


## B.5 — ETL de normalización y purga semántica

Para probar el proceso ETL se preparó intencionalmente el dataset incorporando tres casi-duplicados semánticos: `POL-GAR-DUP-001`, similar a `POL-GAR-001`; `ENV-DUP-001`, similar a `ENV-002`; y `PROC-DUP-001`, similar a `PROC-003`. Además, se agregaron dos inconsistencias estructurales: `FAQ-003` contiene inicialmente la clave `"categoría"` en lugar de `"categoria"`, y `PROC-002` contiene inicialmente `"activo": "true"` como string en lugar de booleano.

El script `etl_purga.py` normaliza nombres de claves, normaliza tipos booleanos y resuelve de forma genérica posibles colisiones de IDs mediante sufijos incrementales, garantizando identificadores únicos sin asumir casos particulares. Luego genera embeddings con el mismo modelo que usan A.4 y B.1 (`gemini-embedding-001`, 768 dimensiones, vía `embeddings_gemini.py`) y reutiliza la función `similitud_coseno` implementada en A.2. La distancia utilizada se calcula como:

`distancia_coseno = 1 - similitud_coseno`

Usar el mismo modelo que el resto de la entrega evita comparar distancias de dos espacios vectoriales distintos: la purga semántica de B.5, el índice FAISS de A.4 y la colección de B.1 quedan sobre un único espacio de embeddings.

Para la purga semántica se calibró `UMBRAL_DISTANCIA = 0.065` a partir de las distancias reales entre todos los pares del dataset. Las distancias de los casi-duplicados intencionales (los "pares de control") fueron:

- `ENV-002` <-> `ENV-DUP-001`: 0.0062
- `POL-GAR-001` <-> `POL-GAR-DUP-001`: 0.0085
- `PROC-003` <-> `PROC-DUP-001`: 0.0278

Los tres pares quedaron por debajo del umbral y se eliminó el documento posterior en cada caso. El siguiente par más cercano en todo el dataset, fuera del catálogo de productos, fue `POL-CAM-003` <-> `POL-CAM-004` con una distancia de 0.1027: dos políticas de cambio genuinamente distintas (cambio de variante del mismo producto vs. cambio por un producto diferente del catálogo), no un duplicado. El umbral se fijó en el punto medio entre 0.0278 (el mayor de los pares de control) y 0.1027 (el par legítimo más cercano), dejando margen para ambos lados.

Durante la calibración se detectó que varios productos legítimos del catálogo tienen distancias coseno muy bajas por ser variantes comerciales con descripciones similares, algunas incluso por debajo del umbral final:

- `PROD-005` <-> `PROD-006`: 0.0493
- `PROD-001` <-> `PROD-002`: 0.0542
- `PROD-009` <-> `PROD-010`: 0.0688
- `PROD-007` <-> `PROD-008`: 0.0842

Estos registros no son duplicados. Por ese motivo, la purga semántica excluye los documentos cuyo ID comienza con `"PROD-"`, ya que una similitud semántica alta entre dos productos no implica que representen el mismo producto — y con el umbral recalibrado esta exclusión sigue siendo necesaria, porque dos de esos pares (0.0493 y 0.0542) caen por debajo de 0.065. Entre los documentos que no pertenecen al catálogo de productos, no se detectaron otros pares legítimos por debajo del umbral.

El resultado del ETL fue:

- Documentos de entrada: 41
- Documentos finales: 38
- Documentos eliminados: `ENV-DUP-001`, `PROC-DUP-001` y `POL-GAR-DUP-001`
- Correcciones estructurales: `FAQ-003`, `"categoría"` -> `"categoria"`; `PROC-002`, `"activo"` de string a booleano

Un `SELECT DISTINCT` no habría detectado estos casos, porque solo identifica registros con valores exactamente iguales. En este dataset los documentos tienen IDs y textos diferentes, aunque expresan prácticamente el mismo concepto. La detección mediante embeddings y distancia coseno permite identificar similitud semántica que no es visible mediante igualdad textual exacta.

## B.6 — Killer Queries

Se diseñaron tres consultas trampa contra `buscar_portalia` (`vector_db.py`), cada una apuntada a un riesgo distinto de un sistema de recuperación semántica: que no entienda jerga sin coincidencia literal, que recomiende algo que en la realidad no se puede vender, y que alucine una respuesta cuando no tiene información. Las tres pasaron con resultados reales, sin ajustar la consulta después de ver que fallaba. La tabla completa, con las distancias reales y la salida cruda de la terminal, está en [`resultados_killer_queries.md`](resultados_killer_queries.md).

En síntesis: la jerga rioplatense ("se me escrachó el celu") se resolvió por semántica pura hacia las políticas de daño/falla sin ninguna coincidencia de palabras; un celular 5G sin stock dejó de recomendarse en cuanto se agregó `--solo-con-stock`, porque el filtro de disponibilidad viaja en el `where` y no como un chequeo posterior; y una consulta ajena al dominio ("alquilar un departamento en Palermo") quedó por encima del umbral de aceptación y el sistema respondió que no tiene esa información, en vez de forzar el resultado más parecido.

# Parte C — Coherencia e Informe

## C.1 — Cadena de coherencia con la Entrega 1

Esta entrega no diseña un sistema nuevo: implementa la Base de Conocimiento que el PEAS de la Entrega 1 (`entrega_1/informe.md`, sección A.3) dejó pendiente y resuelve, con un `where` de ChromaDB, los campos de filtrado que la Matriz de Mapeo de Intenciones (B.3 de la Entrega 1) le asignaba al LLM como parámetros a extraer.

| Elemento de la Entrega 1 | Cómo se implementa en la Entrega 2 |
|---|---|
| Columna "Base de Conocimiento" del PEAS: políticas de cambios, devoluciones y garantías; preguntas frecuentes y procedimientos de postventa; catálogo de productos con descripciones, características y variantes | Los 38 documentos de la colección `portalia_knowledge` (`base_conocimiento_limpia.json`). Los datos dinámicos que el PEAS reservaba para consulta en tiempo real —pedidos, precios reales, estado de envíos— siguen **fuera** de la base vectorial; `en_stock_demo` es solo un metadato de simulación para poder probar el filtrado híbrido de B.4, no una fuente real de stock |
| Campos de filtrado de la Matriz de Mapeo de Intenciones (Entrega 1, B.3): `producto`, `variante_solicitada` para `consultar_stock`; texto libre para `consultar_politica_postventa` | Los metadatos de cada documento: `categoria` agrupa las políticas por tema (cambios, devoluciones, garantías, envíos) y las specs de producto (`red_movil`, `ram_gb`, `tecnologia_panel`, `cancelacion_activa`, `en_stock_demo`, etc.) cubren la variante o característica que el cliente pide |
| Parámetros que el LLM extraía del `texto_libre` (contrato `ExtraccionPostventa` en `parte_c/schemas.py`, Entrega 1) | Ya no se concatenan al prompt como contexto estático: se traducen a condiciones del `where` nativo, por ejemplo `{"$and": [{"categoria": {"$eq": "celulares"}}, {"activo": {"$eq": True}}, {"en_stock_demo": {"$eq": True}}]}` (ver B.4 y B.6/KQ2) |


## C.2 — El umbral de aceptación

El umbral de aceptación es `UMBRAL_ACEPTACION = 0.40` sobre distancia coseno (definido en `vector_db.py`, aplicado en `buscar_portalia`). Se calibró con consultas reales, no con un valor arbitrario: las consultas del dominio de Portalia (políticas de cambio, devolución, garantía) devuelven su mejor coincidencia entre 0.21 y 0.33 de distancia; las consultas ajenas al catálogo (fútbol, recetas, inmuebles) nunca bajan de 0.48. El umbral se fijó en el punto medio de ese rango, con margen para ambos lados (el detalle de la calibración está en `resultados_killer_queries.md`).

Cuando ninguna coincidencia recuperada supera el umbral, `buscar_portalia` no devuelve el resultado más cercano: devuelve `SIN_RESULTADOS` ("No tengo esa información en la base de conocimiento de Portalia.") junto con la mejor distancia encontrada, y una lista de documentos vacía. Esto es intencional: forzar el resultado más cercano cuando nada es realmente relevante es alucinación, el mismo problema que ya se había detectado en la Entrega 1 cuando Gemini inventó el estado y la fecha de entrega del pedido 48327 sin tener ninguna fuente de datos real detrás (`entrega_1/informe.md`, A.2). Responder "no tengo esa información" es la opción correcta, aunque sea menos satisfactoria para el cliente en el momento: es preferible a una respuesta inventada que después hay que desmentir.

## C.3 — Cierre: dónde se conecta

`buscar_portalia` devuelve un `dict` de Python con los IDs, distancias, textos y metadatos de los documentos recuperados (o la respuesta `SIN_RESULTADOS` si ninguno supera el umbral). Eso no es todavía una respuesta para el cliente: falta el **orquestador RAG** (LangChain, Unidad 4) que encadene los pasos que hoy están resueltos por separado en `parte_c/` y `entrega_2/`. Ese orquestador tendría que: (1) tomar la `ExtraccionPostventa` que el LLM ya produce a partir del `texto_libre` del cliente; (2) traducir la intención y los parámetros detectados en el `where` de la búsqueda híbrida (por ejemplo, `consultar_politica_postventa` con `categoria` inferida del mensaje); (3) llamar a `buscar_portalia` con ese filtro; y (4) generar la respuesta final en lenguaje natural usando **exclusivamente** el contexto recuperado, sin agregar información que no esté en los documentos devueltos.

Para las intenciones de riesgo alto ya identificadas en la Entrega 1 (`solicitar_cambio`, `solicitar_devolucion`), el orquestador no puede resolver con RAG solo: antes de redactar cualquier respuesta tiene que pasar por el backend determinista que valida el pedido, el stock y las reglas de negocio (B.4 de la Entrega 1), y solo usar la base vectorial para recuperar el texto de la política aplicable — nunca para decidir si el cambio o la devolución están autorizados.

