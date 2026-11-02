# TP Integrador — Entrega 3
## Sistema RAG con LangChain LCEL y RAG Avanzado

**Dominio:** Portalia (Atención Postventa de E-commerce)  
**Asignatura:** Desarrollo de Sistemas de Inteligencia Artificial  

**Alcance de las evidencias.** Las respuestas, la traza y las métricas conservadas en este
informe corresponden a la versión evaluada de Entrega 3: diez preguntas y el prompt
generador anterior. El mantenimiento amplía el dataset a quince casos y modifica solo
el prompt generador avanzado, sin nuevas consultas a Gemini, evaluaciones RAGAS ni
reindexaciones. Todavía no hay métricas del nuevo prompt o del dataset ampliado.

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
El baseline usado en esta demostración y en la evaluación histórica fija explícitamente
$k=3$ sobre documentos completos (`entrega_3/rag_pipeline.py --modo basico` y
`K_BASICO=3` en el evaluador). El módulo de Parte A tiene otro valor predeterminado,
`K_POR_DEFECTO=10`; no se modificó. La evidencia siguiente corresponde a $k=3$ y no
demuestra que ocurra lo mismo con $k=10$.

#### Caso Testigo:
> *"Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata"*

#### Evidencia del RAG Básico ($k=3$):
Al ejecutar la consulta en el RAG básico, ChromaDB recuperó:
1. `POL-DEV-001`: Plazo de devolución (10 días).
2. `POL-CAM-002`: Verificaciones del estado del producto y sus accesorios para cambios.
3. `POL-CAM-004`: Gestión de diferencia de precio al cambiar por otro producto del catálogo.

**Respuesta generada por el LLM:**
* Informa que la devolución ya no es posible porque el plazo de 10 días venció, aunque el usuario indicó la fecha de compra y no la de recepción.
* Indica consultar precios y disponibilidad e informar la diferencia de precio para el cambio.
* ⚠️ **Falla / Omisión crítica:** El asistente **no le responde al cliente si a los 15 días puede realizar el cambio**. Omitió por completo el plazo general para cambios.

#### Causa Técnica:
* **Cobertura insuficiente del top-$k$:** entre las políticas complementarias necesarias para responder la consulta, faltó `POL-CAM-001`, que establece el plazo de **30 días corridos desde la recepción** para solicitar cambios. La selección de tres documentos por similitud dejó esa política fuera del contexto; no fue un dato presente que el generador ignorara.
* **Clasificación según la consigna:** el caso se aproxima a **Ruido en el contexto** por la competencia entre información relacionada al seleccionar el contexto. Es una aproximación: los tres documentos recuperados eran pertinentes, por lo que no se demuestra que hubiera documentos irrelevantes. La descripción técnica más precisa es una falla de recuperación por cobertura insuficiente.
* **Categorías que no demuestra la evidencia:** no hubo **Chunk cortado**, porque se recuperaban documentos completos; tampoco **Lost in the Middle**, porque la política omitida nunca llegó al contexto. No corresponde atribuir el fallo a su posición dentro del prompt.
* **Límite adicional de la respuesta:** no mencionar un plazo ausente evita inventarlo, pero no hace correcta toda la respuesta: la conclusión sobre devolución también requiere conocer cuándo se recibió el producto.

---

### B.2. Reindexación con Solapamiento (Chunking con Overlap)
En la implementación avanzada original se aplicó un preprocesamiento con `RecursiveCharacterTextSplitter`:
* **Configuración:** `chunk_size = 300` caracteres, `chunk_overlap = 80` caracteres.
* **Separadores semánticos:** `[". ", "\n\n", "\n", ", ", " "]`.
* **Comparativa de volumen:**
  * **Documentos originales monolíticos:** 38 documentos.
  * **Chunks generados con solapamiento:** **57 chunks**.

El solapamiento reduce la pérdida de continuidad entre bloques, pero no garantiza que
toda regla o excepción quede completa en cada chunk. Además, recuperar solo algunos
chunks puede dejar fuera condiciones presentes en otros fragmentos del mismo documento.

**¿Resuelve la falla de B.1?** El chunking por sí solo no asegura mayor cobertura.
La mejora observada combina **chunking, recuperación amplia de ocho candidatos y
reranking**. En la demostración esa combinación incorporó el plazo omitido; no se
aisló el efecto de cada componente ni se probó una resolución completa de C1.

---

### B.3. Reranking en Dos Etapas (LLM como Juez)
Se implementó un pipeline en dos etapas para desacoplar la recuperación de candidatos de la generación final:

1. **Etapa 1 — Recuperación Amplia ($k=8$ candidatos):**  
   ChromaDB recupera los 8 fragmentos más cercanos por similitud coseno. Al ampliar la ventana a 8, la política omitida `POL-CAM-001_chk1` (plazo de cambios) ingresa exitosamente en la posición 7 de los candidatos.

2. **Etapa 2 — Filtrado y Scoring con LLM Juez:**  
   Un prompt intermedio especializado con Gemini ($T=0$) recibe la consulta y los 8 chunks candidatos. Evalúa su aporte para responder y selecciona hasta 3 fragmentos. En el mantenimiento se respetan selecciones vacías y se controlan IDs únicos, conocidos y el límite máximo, sin cambiar el prompt del juez.
   * **Chunks seleccionados por el Reranker:**
     * `POL-CAM-001_chk1`: Plazo general para cambios (30 días).
     * `POL-CAM-004_chk1`: Primer fragmento de la política de cambio por otro producto.
     * `POL-DEV-001_chk1`: Plazo para devoluciones (10 días).

---

#### B.3.1. Mejora Observada y Límites de la Resolución
En la demostración original del pipeline avanzado (`entrega_3/rag_pipeline.py --modo avanzado`) se registró esta respuesta. Se conserva como evidencia histórica, no como salida del nuevo prompt:

> *"Sobre tu consulta, te informo lo siguiente según nuestras políticas:*  
> * ***Cambios:*** *Puedes solicitar un cambio por otro artículo del catálogo de Portalia dentro de los 30 días corridos posteriores a la recepción de tu pedido. Para iniciar el proceso, debes identificar el pedido y el producto correspondiente.*  
> * ***Devoluciones:*** *El plazo para solicitar una devolución es de 10 días corridos posteriores a la recepción del producto.*  
>  
> ***Dado que mencionas haber recibido los auriculares hace 15 días, el plazo para solicitar una devolución ya ha expirado, pero aún te encuentras dentro del periodo habilitado para gestionar un cambio por otro producto.***"*

**Alcance:** se recuperó y mencionó el plazo general de cambios que faltaba en el
baseline. Sin embargo, **C1 no quedó completamente resuelta**: la pregunta dice
"compré hace 15 días", no "recibí hace 15 días". Hace falta pedir la fecha de recepción
y explicar los plazos de manera condicional. También falta explicar la gestión de la
diferencia de precio solicitada, con respaldo documental y sin inventar importes.
Tener el ID de una política entre las fuentes no prueba que se hayan recuperado y
explicado todas sus condiciones. Estas limitaciones también deben revisarse en la
respuesta C1 guardada en la corrida RAGAS.

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
  * *Impacto observado:* reducción de 8 candidatos a 3 fragmentos e incorporación de `POL-CAM-001_chk1` (plazo de 30 días). Esto mejora la cobertura del plazo, sin demostrar que toda la consulta esté resuelta ni que tres fragmentos sean una ventana óptima.


---

## Parte C — Evaluación Cuantitativa con RAGAS

Script: [`evaluacion_ragas.py`](evaluacion_ragas.py). Los resultados históricos por
pregunta y pipeline están en [`resultados_ragas.json`](resultados_ragas.json), que se
conserva intacto. Corresponden a diez preguntas y al prompt anterior, sobre bases recién
cargadas y sin evento en caliente. Las tablas C.2 y C.3 describen únicamente esa corrida.

**Metodología.**
* **Pipelines evaluados:** RAG básico (colección original, `k=3`, el mismo de B.1) y RAG avanzado (chunking 300/80, `k=8` candidatos, LLM juez que selecciona 3).
* **Juez de RAGAS (0.4.3):** Gemini consumido por su endpoint compatible con OpenAI (el cliente async de `google-genai` falla con RAGAS 0.4), con `gemini-embedding-001` para Answer Relevancy (`strictness=1`).
* **Preguntas de escape:** no se puntúan con RAGAS. El evaluador histórico detectaba la frase como subcadena, sin exigir coincidencia exacta. El evaluador mantenido registra por separado abstención exacta y coincidencia parcial. Los promedios históricos excluyen los dos casos de escape y usan **n=8**, no el dataset actual de quince casos.

### C.1. Golden Dataset

El dataset actual contiene **15 preguntas: 5 simples, 5 complejas, 2 de escape y
3 informales**. Las diez preguntas siguientes, sus respuestas esperadas y sus
documentos de referencia se conservaron sin modificaciones:

| ID | Tipo | Pregunta | Documentos de referencia |
|---|---|---|---|
| S1 | simple | ¿Cuántos días tengo para pedir un cambio de producto? | POL-CAM-001 |
| S2 | simple | Mi pedido figura como despachado, ¿me llega hoy? | ENV-003 |
| S3 | simple | ¿Qué datos tengo que dar para iniciar una devolución? | FAQ-002 |
| C1 | compleja | Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata | POL-CAM-001, POL-CAM-004, POL-DEV-001 |
| C2 | compleja | Me aceptaron la devolución, ¿qué verificaron y ya tengo el reintegro? | POL-DEV-002, POL-DEV-003 |
| C3 | compleja | La notebook me llegó con la caja mojada y además no prende, ¿qué hago? | POL-GAR-001, POL-GAR-002 |
| E1 | escape | ¿Aceptan pago en cuotas con tarjeta de crédito? | — |
| E2 | escape | ¿Cuánto cuesta el televisor Vista 50 QLED? | — |
| I1 | informal | che el auri se me rompió a la semana, me lo cambian o q onda | POL-GAR-001 |
| I2 | informal | quiero el mismo celu pero en otro color, se puede? | POL-CAM-003 |

Las respuestas esperadas completas están en `GOLDEN_DATASET`. Los cinco casos nuevos
se verificaron contra `entrega_2/base_conocimiento_limpia.json`, incluyendo metadatos
cuando corresponden, y todavía no tienen resultados RAGAS:

| ID / tipo | Pregunta nueva | Resumen de la respuesta esperada | Documentos de referencia |
|---|---|---|---|
| S4 / simple | ¿El monitor Detail 27 USB-C permite transmitir video y cargar una notebook con un solo cable? ¿Qué hay que verificar? | Sí, para una notebook compatible; antes de recomendar la carga se verifica la potencia requerida por el equipo. | PROD-018 |
| S5 / simple | ¿Cuándo se deriva una consulta a un operador humano y qué información debe acompañarla? | Ante identificación insegura, información indispensable que el cliente no puede aportar o decisión manual; incluir motivo, datos verificados y contexto, sin afirmar aprobación. | PROC-001 |
| C4 / compleja | Quiero cambiar por otro producto, pero todavía no me informaron la diferencia de precio ni acepté los términos. ¿Pueden registrar el cambio igual? | No; verificar pedido, artículo, condiciones, disponibilidad y precios, informar diferencia y gestión, y obtener aceptación antes de ejecutar. | POL-CAM-004, PROC-002 |
| C5 / compleja | El Nova 5G 256 figura sin stock de demostración. ¿Eso significa que hoy no puedo comprarlo? ¿Cómo se confirma? | El indicador demo en falso no determina stock real; consultar inventario del modelo y variante, sin fingir una consulta realizada. | PROD-003, FAQ-003, PROC-003 |
| I3 / informal | che, ya pasó la fecha de entrega y el correo no actualiza nada, ¿dónde está el paquete y cuándo cae? | Identificar el pedido y consultar backend y última información logística; sin confirmación, no inventar ubicación, causa ni fecha. | ENV-001, ENV-002 |

La ampliación cubre compatibilidad técnica, derivación humana, aceptación previa a
operaciones, límites del stock demo y demora logística con lenguaje informal. En los
casos de escape el esperado sigue siendo la frase exacta del pipeline.

**Limitación conservada en C1:** su referencia histórica aplica los plazos como si
los quince días fueran desde la recepción, aunque la pregunta solo menciona la compra.
Se mantiene por el requisito de no modificar los diez casos originales. La próxima
evaluación debe registrar esta ambigüedad y revisar semánticamente C1: una respuesta
condicional que pida la fecha de recepción puede ser correcta aunque difiera de esa
referencia. Su puntaje automático aislado no basta para juzgar el nuevo prompt.

### C.2. Línea Base Histórica (RAG Básico, Diez Casos)

| ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| S1 | 1.00 | 0.82 | 1.00 | 1.00 |
| S2 | 1.00 | 0.93 | 1.00 | 1.00 |
| S3 | 1.00 | 0.73 | 1.00 | 1.00 |
| C1 | 0.50 | 0.75 | 0.00 | 0.67 |
| C2 | 1.00 | 0.65 | 0.50 | 1.00 |
| C3 | 0.70 | 0.88 | 1.00 | 1.00 |
| I1 | 1.00 | 0.60 | 1.00 | 1.00 |
| I2 | 1.00 | 0.79 | 1.00 | 1.00 |
| **Promedio (n=8)** | **0.900** | **0.768** | **0.812** | **0.958** |

En C1 falta `POL-CAM-001` entre las fuentes del básico (`POL-DEV-001`, `POL-CAM-002`,
`POL-CAM-004`), lo que demuestra cobertura incompleta. El juez asignó Context Precision
0.00 y Recall 0.67; esos puntajes no demuestran que los tres documentos fueran irrelevantes.

### C.3. Comparación Histórica: Básico vs. Avanzado

| Métrica | Básico | Avanzado | Delta |
|---|---|---|---|
| Faithfulness | 0.900 | 0.890 | −0.010 |
| Answer Relevancy | 0.768 | 0.645 | −0.123 |
| Context Precision | 0.812 | 0.937 | **+0.125** |
| Context Recall | 0.958 | 0.958 | 0.000 |

El evaluador histórico marcó correctos los dos escapes en ambos pipelines por
coincidencia de subcadena. Con el contrato de igualdad exacta, solo E1 del básico
coincide: E1 avanzado agrega un punto y E2 agrega explicaciones en ambos pipelines.
Esta revisión no modifica los flags ni los puntajes del JSON histórico. Ninguna de las
ocho preguntas restantes fue marcada como abstención por aquel evaluador.

**La corrida registra mayor Context Precision (+0.125).** En C1 el avanzado incorpora
`POL-CAM-001_chk1` y el puntaje pasa de 0.00 a 1.00. Esto es consistente con la mejora
de cobertura observada en B.3.1, sin probar ausencia de omisiones ni aislar el efecto
del chunking, la recuperación amplia o el reranker.

**Los deltas negativos no son uniformes.** Con n=8 pesan especialmente algunos casos:
* Answer Relevancy cae sobre todo en S2 (0.93 → 0.00), que agrega *"No poseo información sobre la fecha de entrega de tu pedido"*. Una posible lectura como evasión es una hipótesis; el JSON no conserva la explicación interna del juez y no permite confirmarla.
* En C1 Faithfulness pasa de 0.50 a 0.29. No corresponde atribuirlo exclusivamente a una penalización injustificada de inferencias: la respuesta asume una fecha de recepción que el cliente no dio. Deben revisarse las afirmaciones y el contexto exacto antes de concluir sobre la fidelidad. La omisión de la diferencia de precio es, además, un problema de completitud; por sí sola no explica un puntaje de Faithfulness.
* Context Recall no se mueve (0.958 en ambos): los dos recuperan la información clave, salvo la que se pierde en C1 en ambos casos.

### C.4. Diagnóstico de la Métrica más Baja

La métrica más baja del RAG avanzado en la corrida histórica es **Answer Relevancy
(0.645)**. Ese valor no mide el prompt mantenido ni el dataset de quince preguntas.

**Dónde se deteriora.** Los mayores descensos de Answer Relevancy guardados son:

| Caso | Básico | Avanzado | Diferencia |
|---|---|---|---|
| S2: despachado | 0.930 | 0.000 | −0.930 |
| C3: embalaje mojado y falla | 0.877 | 0.680 | −0.197 |
| I2: otro color | 0.794 | 0.765 | −0.028 |

S2 explica aproximadamente el 95% de la caída promedio neta. C2 e I1 mantienen valores
bajos, pero mejoran ligeramente. Las ocho respuestas avanzadas puntuadas son más cortas
que las básicas; no se puede explicar la caída únicamente por extensión.

**Hipótesis y evidencia textual.** El prompt anterior pedía, además de lo solicitado,
incluir condiciones y evaluaciones documentadas. Esa obligación favorece información
lateral: I1 termina hablando de falta de información sobre el catálogo, algo no pedido;
S3 agrega validaciones a una pregunta sobre datos para iniciar el trámite. En S2 aparece
un descargo sobre la fecha, aunque sí hay información para explicar qué significa
"despachado". Estos ejemplos justifican enfocar el prompt, pero no prueban por sí solos
la causa interna de cada puntaje del juez ni garantizan una mejora cuantitativa.

**Cambio aplicado, pendiente de evaluación real.** El generador avanzado ahora usa
`REGLAS_GENERADOR_AVANZADO`, propio de `parte_b/rag_avanzado.py`. Conserva las reglas de
respaldo documental, corrección de premisas falsas, protección ante instrucciones en
los datos y límites sobre operaciones, stock y estados reales. La Parte A y el prompt
del reranker permanecen intactos. El nuevo alcance exige:

1. Responder primero la pregunta concreta y todos sus aspectos respaldados.
2. Incluir solo condiciones indispensables, evitando información lateral no solicitada.
3. Distinguir compra de recepción; si falta la fecha de recepción, explicar plazos
   condicionalmente y pedirla, sin concluir que vencieron o siguen habilitados.
4. Responder lo respaldado de una consulta parcial y señalar el dato concreto faltante,
   sin insertar la frase de escape como descargo parcial.
5. Si ninguna parte solicitada puede responderse, devolver únicamente la frase de
   escape exacta, sin puntuación ni explicaciones adicionales.

**Validación y próximos pasos.** Las pruebas locales verifican el dataset, sus fuentes,
el prompt enviado al doble del generador y las regresiones de seguridad. No verifican
que Gemini obedezca siempre esas instrucciones. No se ejecutaron consultas reales ni
RAGAS durante este mantenimiento y no se reindexó ninguna colección.

La evaluación real deberá revisar primero S2, C3, I1 y C1, incluyendo fecha de recepción,
diferencia de precio, fidelidad y abstenciones exactas. Después podrá evaluar ambos
pipelines sobre los mismos quince casos, con bases y configuración controladas, en un
experimento nuevo según el [README](../README.md#evaluación-con-ragas-parte-c). No se deben
mezclar parciales previos ni comparar directamente el promedio de quince preguntas con
el histórico de diez como si la diferencia aislara el efecto del prompt. La ambigüedad
de la referencia C1 debe constar en la revisión semántica.

**Limitaciones históricas.** La corrida puntúa solo ocho preguntas y usa un LLM como
juez con `strictness=1` para Answer Relevancy. Las diferencias pequeñas no son
concluyentes. El informe original también consignó un recálculo exploratorio de
Faithfulness de C1 (básico 0.80 y avanzado 0.29); no se repitió aquí ni reemplaza los
valores 0.50 y 0.29 de la corrida guardada. Todas las tablas históricas conservan los
valores de `resultados_ragas.json`, correspondientes a bases sin evento en caliente.
