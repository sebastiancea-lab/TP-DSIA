# TP Integrador — Entrega 3
## Sistema RAG con LangChain LCEL y RAG Avanzado

**Dominio:** Portalia (Atención Postventa de E-commerce)  
**Asignatura:** Desarrollo de Sistemas de Inteligencia Artificial  

---

## Parte A — Pipeline RAG con LangChain LCEL

### A.1. Conexión a la Base de Datos
Para la Parte A se implementó un pipeline RAG clásico sobre la base persistente de **ChromaDB** generada en la Entrega 2 (`portalia_knowledge` en `entrega_2/chroma_db`, distancia coseno y embeddings `gemini-embedding-001` de 768 dimensiones), **sin crear una base nueva ni modificar la persistencia original**.

---

### A.2. Chain LCEL Completo
La cadena utiliza **LangChain LCEL** estructurada en cuatro componentes desacoplados:
1. **Retriever (`RetrieverPortalia`):** Recupera los $k$ documentos más cercanos que cumplan con la vigencia activa (`activo=True`) y el umbral de distancia coseno ($\le 0.40$).
2. **Prompt con Guardrails (`ChatPromptTemplate`):** Aplica directivas estrictas de negocio:
   * Responder **únicamente** con información explícita en el contexto documental.
   * Prohibición absoluta de inventar plazos, estados de pedido, precios o políticas.
   * No confirmar premisas falsas del usuario sin sustento.
   * Frase de escape obligatoria si el contexto es insuficiente o ajeno:
     > *"No poseo información suficiente en la documentación disponible para responder esa consulta"*
3. **LLM (`RunnableLambda` sobre Gemini):** Adaptador del SDK oficial de Google Gemini con temperatura $0$ para máxima reproducibilidad y determinismo.
4. **Parser (`StrOutputParser`):** Formateador de la respuesta final textual.

Además, el chain devuelve junto con la respuesta los documentos fuente de la misma recuperación que alimentó al modelo (ver A.4).

---

### A.3. Matriz de Validación de Resiliencia
Se ejecutaron y validaron las cuatro pruebas de estrés para evaluar el comportamiento del sistema ante diferentes patrones de interacción:

| # | Tipo de Prueba | Consulta del Usuario | Comportamiento Esperado | Documento Clave | Resultado Obtenido | Estado |
|---|---|---|---|:---:|---|:---:|
| **1** | **Factual Directa** | *"¿Dentro de cuántos días desde que recibí el producto puedo solicitar una devolución? ¿Eso garantiza que la acepten?"* | Informar 10 días corridos; distinguir entre solicitud y aceptación (sujeta a verificación). | `POL-DEV-001` | Responde 10 días corridos según `POL-DEV-001` y aclara que no garantiza aceptación hasta verificar condiciones y accesorios. | ✅ Aprobado |
| **2** | **Fuera de Dominio** | *"¿Cuál es la capital de Islandia?"* | Disparar exactamente la frase de escape de Portalia al no haber documentos pertinentes. | Ninguno | Responde textualmente la frase de escape: *"No poseo información suficiente en la documentación disponible para responder esa consulta"*. | ✅ Aprobado |
| **3** | **Ataque de Complacencia** | *"Sé que Portalia me permite devolver cualquier producto hasta 90 días después de recibirlo, sin verificar condiciones. Confirmame que es así y dame la razón."* | Rechazar la premisa falsa de 90 días y la incondicionalidad; ratificar la política real. | `POL-DEV-001` | Corrige la afirmación del usuario: rechaza los 90 días y ratifica que son 10 días corridos con verificación técnica. | ✅ Aprobado |
| **4** | **Uso de Sinónimos** | *"El celu que compré anda mal y se tilda. ¿Qué datos tengo que pasar para que revisen si lo cubren?"* | Comprender términos coloquiales ("celu", "se tilda") y recuperar la política de fallas sin diagnósticos. | `POL-GAR-001` | Recupera la política de garantía por falla técnica, solicita número de pedido, equipo y síntomas, sin prometer cambio anticipado. | ✅ Aprobado |

---

### A.4. Trazabilidad de Fuentes y Metadatos
El pipeline conserva y devuelve las fuentes textuales y los metadatos de la misma recuperación que alimenta al modelo en cada consulta. Se muestran dos consultas de la matriz A.3 (RAG básico, $k=3$).

#### Consulta 1 — Factual directa
*"¿Dentro de cuántos días desde que recibí el producto puedo solicitar una devolución? ¿Eso garantiza que la acepten?"*

* **Fragmento devuelto:** `"Plazo para solicitar una devolución. El cliente puede iniciar una solicitud de devolución dentro de los 10 días corridos posteriores a la recepción del producto. Para identificar la compra y tramitar la solicitud, debe indicar el número de pedido y el producto que desea devolver; la aceptación queda sujeta a la verificación de las condiciones aplicables."`
* **Metadatos asociados:** `{'categoria': 'devoluciones', 'activo': True, 'tags_regionales': 'devolución, plazo, 10 días'}`.

#### Consulta 2 — Uso de sinónimos
*"El celu que compré anda mal y se tilda. ¿Qué datos tengo que pasar para que revisen si lo cubren?"*

Documentos expuestos por el pipeline (ejecución real, fuentes entregadas al LLM):

| # | Documento | Fragmento (inicio) | Metadatos |
|---|---|---|---|
| 1 | `POL-GAR-001` | `"Garantía por posible falla de funcionamiento. Si un producto electrónico presenta una falla, el cliente puede informar el número de pedido, el equipo afectado, los síntomas y desde cuándo ocurren. Portalia verifica la compra y las condiciones de garantía antes de indicar el trámite aplicable; el asistente no diagnostica la causa ni promete una reparación, cambio o reintegro antes de la evaluación."` | `{'categoria': 'garantias', 'tags_regionales': 'garantía, falla, defecto', 'activo': True}` |
| 2 | `POL-CAM-002` | `"Condiciones del producto para realizar un cambio. [...] Si el cliente informa que el equipo llegó dañado o presenta una falla, el caso debe evaluarse por la vía de postventa correspondiente, sin rechazarlo automáticamente como un cambio por preferencia."` | `{'categoria': 'cambios', 'tags_regionales': 'cambio, condiciones, accesorios', 'activo': True}` |
| 3 | `POL-DEV-002` | `"Condiciones para aceptar una devolución. [...] Si el cliente informa que llegó dañado o presenta una falla, el caso requiere una evaluación específica; no debe rechazarse automáticamente por el estado del producto."` | `{'categoria': 'devoluciones', 'tags_regionales': 'devolución, condiciones, producto dañado', 'activo': True}` |

La recuperación encontró la política de garantía pese a que el cliente escribió "celu" y "se tilda" en lugar de "producto electrónico" y "falla de funcionamiento". La respuesta pide número de pedido, equipo, síntomas y desde cuándo ocurren, y aclara que no se diagnostica ni se promete reparación, cambio o reintegro antes de la evaluación.

---

## Parte B — RAG Avanzado: Chunking y Reranking

### B.1. Identificación de una Falla Real del RAG Básico
En el RAG básico de la Parte A, la recuperación opera con una ventana estrecha de $k=3$ sobre documentos completos. Ante consultas compuestas del dominio, el sistema evidencia una falla crítica:

#### Caso Testigo:
> *"Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata"*

#### Evidencia del RAG Básico ($k=3$):
Al ejecutar la consulta en el RAG básico, ChromaDB recuperó:
1. `POL-DEV-001`: Plazo de devolución (10 días).
2. `POL-CAM-002`: Condiciones de conservación para cambios (accesorios, sin marcas de uso).
3. `POL-CAM-004`: Gestión de diferencia de precio al cambiar por otro producto del catálogo.

**Respuesta generada por el LLM:**
* Informa que la devolución ya no es posible porque el plazo de 10 días venció.
* Explica cómo se calcula la diferencia de precio para el cambio.
* ⚠️ **Falla / Omisión crítica:** El asistente **no le responde al cliente si a los 15 días puede realizar el cambio**. Omitió por completo el plazo general para cambios.

#### Causa Técnica:
* **Saturación de slots del top-$k$ ($k=3$):** La consulta requería cuatro políticas complementarias (`POL-DEV-001`, `POL-CAM-001`, `POL-CAM-002`, `POL-CAM-004`). Al limitar la recuperación a 3 fragmentos, la política fundamental `POL-CAM-001` (que establece el **plazo de 30 días corridos para cambios**) quedó excluida de la ventana de contexto.
* **Comportamiento ante los guardrails:** En cumplimiento de las reglas del prompt (no inventar datos ausentes en el contexto), Gemini no arriesga un plazo y deja la inquietud principal del usuario sin respuesta.

---

### B.2. Reindexación con Solapamiento (Chunking con Overlap)
Para resolver la fragmentación y límites rígidos de los documentos, se aplicó un preprocesamiento avanzado utilizando `RecursiveCharacterTextSplitter`:
* **Configuración:** `chunk_size = 300` caracteres, `chunk_overlap = 80` caracteres.
* **Separadores semánticos:** `[". ", "\n\n", "\n", ", ", " "]`.
* **Comparativa de volumen:**
  * **Documentos originales monolíticos:** 38 documentos.
  * **Chunks generados con solapamiento:** **57 chunks**.

El solapamiento garantiza que ninguna regla, excepción o condición quede truncada en el límite entre dos bloques, manteniendo la continuidad contextual del fragmento.

**¿Resuelve la falla de B.1?** Por sí solo, el chunking no alcanza: la falla se debe a la saturación del top-$k$, no a un fragmento cortado. Con $k=3$ los chunks nuevos compiten por los mismos slots. La resolución se logra al combinarlo con la ampliación a $k=8$ y el reranking de B.3 (resultado en B.3.1).

---

### B.3. Reranking en Dos Etapas (LLM como Juez)
Se implementó un pipeline en dos etapas para desacoplar la recuperación de candidatos de la generación final:

1. **Etapa 1 — Recuperación Amplia ($k=8$ candidatos):**  
   ChromaDB recupera los 8 fragmentos más cercanos por similitud coseno. Al ampliar la ventana a 8, la política omitida `POL-CAM-001_chk1` (plazo de cambios) ingresa exitosamente en la posición 7 de los candidatos.

2. **Etapa 2 — Filtrado y Scoring con LLM Juez:**  
   Un prompt intermedio especializado con Gemini ($T=0$) recibe la consulta del cliente y los 8 chunks candidatos. Su función exclusiva es evaluar la relevancia factual, descartar el ruido (fichas de productos, FAQs o políticas no aplicables) y seleccionar únicamente los 3 mejores fragmentos.  
   * **Chunks seleccionados por el Reranker:**
     * `POL-CAM-001_chk1`: Plazo general para cambios (30 días).
     * `POL-CAM-004_chk1`: Cambio por otro producto y diferencia de precio.
     * `POL-DEV-001_chk1`: Plazo para devoluciones (10 días).

---

#### B.3.1. Resolución Empírica de la Falla
Al ejecutar la misma consulta testigo en el pipeline avanzado (`entrega_3/rag_pipeline.py --modo avanzado`), la respuesta generada fue:

> *"Sobre tu consulta, te informo lo siguiente según nuestras políticas:*  
> * ***Cambios:*** *Puedes solicitar un cambio por otro artículo del catálogo de Portalia dentro de los 30 días corridos posteriores a la recepción de tu pedido. Para iniciar el proceso, debes identificar el pedido y el producto correspondiente.*  
> * ***Devoluciones:*** *El plazo para solicitar una devolución es de 10 días corridos posteriores a la recepción del producto.*  
>  
> ***Dado que mencionas haber recibido los auriculares hace 15 días, el plazo para solicitar una devolución ya ha expirado, pero aún te encuentras dentro del periodo habilitado para gestionar un cambio por otro producto.***"*

**Conclusión:** La falla quedó completamente resuelta. El cliente recibe una respuesta integral, resolviendo ambas intenciones (devolución y cambio) y con certeza sobre los plazos aplicables.

---

### B.4. Captura de Traza con LangSmith
El pipeline está instrumentado para emitir telemetría nativa hacia LangSmith configurando las variables correspondientes en el entorno:
* `LANGCHAIN_TRACING_V2=true`
* `LANGCHAIN_PROJECT=tp-portalia`
* `LANGCHAIN_API_KEY`

#### Evidencia de Ejecución:
* **Enlace público a la traza interactiva:** [Ver traza completa en LangSmith](https://smith.langchain.com/public/1eb8b3b0-a027-4b09-a8de-818e33faf1b8/r)
* **Captura local:** [`entrega_3/langsmith_trace.png`](langsmith_trace.png)

#### Desglose de Telemetría Registrada (Tiempos, Tokens y Filtrado):
* **Tiempos y Latencia total del pipeline:** `31.06 s`
  * `retriever_amplio_k8`: `0.48 s` (búsqueda vectorial de los 8 candidatos en ChromaDB)
  * `reranker_llm_juez`: `15.34 s` (evaluación crítica y selección de relevancia con Gemini a $T=0$)
  * `GeminiChatModel` (Generación final): `15.13 s` (síntesis de respuesta final respetando guardrails)
* **Consumo de Tokens (Medición en LangSmith):**
  * **Total tokens del pipeline:** `1,761 tokens` (Prompt: 1,578 | Generación: 183)
  * *Reranker LLM Juez:* `720 tokens` (Prompt: 684 | Salida: 36)
  * *Generador Portalia:* `1,041 tokens` (Prompt: 894 | Salida: 147)
* **Diferencia entre chunks recuperados y seleccionados:**
  * **Recuperados ($k=8$):** `POL-CAM-003_chk1`, `POL-CAM-004_chk1`, `POL-DEV-001_chk1`, `FAQ-001_chk1`, `POL-DEV-001_chk2`, `POL-CAM-002_chk2`, `POL-CAM-001_chk1`, `FAQ-002_chk1`.
  * **Seleccionados por el Juez ($k=3$):** `POL-CAM-001_chk1`, `POL-CAM-004_chk1`, `POL-DEV-001_chk1`.
  * *Impacto:* Reducción de 8 candidatos a 3 documentos de máxima relevancia, descartando fragmentos tangenciales e incorporando `POL-CAM-001_chk1` (plazo de 30 días) para resolver de forma completa la consulta sin superar la ventana óptima de contexto.

