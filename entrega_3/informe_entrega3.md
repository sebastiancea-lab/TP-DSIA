# TP Integrador — Entrega 3
## Sistema RAG con LangChain LCEL y RAG Avanzado

**Dominio:** Portalia (Atención Postventa de E-commerce)  
**Asignatura:** Desarrollo de Sistemas de Inteligencia Artificial  

**Alcance de las evidencias.** Las métricas de la Parte C (C.2 a C.4) corresponden a una
evaluación RAGAS nueva con quince preguntas y el prompt generador avanzado actualizado
(`entrega3-15casos`, guardada en `resultados_ragas_15casos.json`). La corrida histórica de diez
preguntas y el prompt anterior se conserva como anexo. La demostración de la Parte B (B.3.1),
las respuestas de ejemplo y la traza de LangSmith (B.4) provienen de la versión original y no
se volvieron a generar. No se reindexó ninguna colección.

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
* **Categoría de la consigna: Ruido en el contexto.** Con $k=3$ el contexto tiene solo tres lugares y los ocupan documentos parecidos a la consulta pero secundarios para su duda central. `POL-DEV-001` y `POL-CAM-004` sí aportan, pero `POL-CAM-002` (verificaciones del estado del producto y sus accesorios) compite con `POL-CAM-001`, la política indispensable: el plazo de **30 días corridos desde la recepción** para solicitar cambios. El contenido menos útil desplaza al que responde la pregunta principal, y el generador, que no puede usar lo que no recibe, deja esa parte sin respuesta.
* **Evidencia del ruido:** el contexto recuperado no incluye `POL-CAM-001`, la respuesta omite el plazo de cambios y el juez RAGAS asignó a este caso Context Precision 0.00 y Context Recall 0.67 (ver C.2).
* **Categorías descartadas:** no es **Chunk cortado**, porque en el básico se recuperan documentos completos; tampoco **Lost in the Middle**, porque la política omitida nunca llegó al contexto, de modo que su posición en el prompt no interviene.
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

**¿Resuelve la falla de B.1 (ruido en el contexto con $k=3$)?** El chunking por sí solo no asegura mayor cobertura.
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

Script: [`evaluacion_ragas.py`](evaluacion_ragas.py). Los resultados por pregunta y pipeline de
la evaluación vigente (quince preguntas) están en
[`resultados_ragas_15casos.json`](resultados_ragas_15casos.json). La corrida histórica de diez
preguntas y el prompt anterior, en [`resultados_ragas.json`](resultados_ragas.json), se conserva
intacta y sus tablas están en el Anexo C. Ambas se hicieron sobre bases sin evento en caliente.

**Metodología.**
* **Pipelines evaluados:** RAG básico (colección original, `k=3`, el mismo de B.1) y RAG avanzado (chunking 300/80, `k=8` candidatos, LLM juez que selecciona 3).
* **Juez de RAGAS (0.4.3):** Gemini consumido por su endpoint compatible con OpenAI (el cliente async de `google-genai` falla con RAGAS 0.4), con `gemini-embedding-001` para Answer Relevancy (`strictness=1`).
* **Preguntas de escape:** no se puntúan con RAGAS. El evaluador histórico detectaba la frase como subcadena, sin exigir coincidencia exacta. El evaluador actual registra por separado abstención exacta y coincidencia parcial, y solo la igualdad exacta aprueba un caso de escape. Los promedios vigentes excluyen los dos casos de escape y usan **n=13**; los históricos usan n=8.
* **Variabilidad del juez:** el juez es un LLM no determinista. Un pipeline sin cambios (el básico) obtuvo Answer Relevancy distinta entre dos corridas, por lo que las diferencias de pocas centésimas no son concluyentes (ver C.4).

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
cuando corresponden, y se evaluaron en la corrida vigente (C.2 y C.3):

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
Se mantiene por el requisito de no modificar los diez casos originales. En la corrida vigente
el avanzado responde a C1 de forma condicional y pide la fecha de recepción, lo que puede ser
correcto aunque difiera de esa referencia; por eso su puntaje automático aislado no basta para
juzgar el prompt nuevo (ver C.3 y C.4).

### C.2. Línea Base (RAG Básico, Quince Casos)

Corrida `entrega3-15casos`, con el dataset completo de quince preguntas. El detalle por
caso y pipeline está en [`resultados_ragas_15casos.json`](resultados_ragas_15casos.json).

| ID | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| S1 | 1.00 | 0.82 | 1.00 | 1.00 |
| S2 | 1.00 | 0.93 | 1.00 | 1.00 |
| S3 | 1.00 | 0.77 | 1.00 | 1.00 |
| S4 | 1.00 | 0.88 | 1.00 | 1.00 |
| S5 | 1.00 | 0.94 | 1.00 | 1.00 |
| C1 | 0.80 | 0.74 | 0.00 | 0.67 |
| C2 | 1.00 | 0.00 | 0.50 | 1.00 |
| C3 | 0.78 | 0.82 | 1.00 | 1.00 |
| C4 | 0.71 | 0.81 | 1.00 | 1.00 |
| C5 | 1.00 | 0.65 | 0.50 | 1.00 |
| I1 | 1.00 | 0.59 | 1.00 | 1.00 |
| I2 | 1.00 | 0.76 | 1.00 | 1.00 |
| I3 | 1.00 | 0.00 | 1.00 | 1.00 |
| **Promedio (n=13)** | **0.946** | **0.670** | **0.846** | **0.974** |

Los dos casos de escape (E1, E2) no se puntúan con RAGAS, por eso cada promedio sale de
**n=13**. En C1 el básico vuelve a omitir `POL-CAM-001` (contexto `POL-DEV-001`,
`POL-CAM-002`, `POL-CAM-004`) y repite Context Precision 0.00 y Recall 0.67, igual que en la
corrida histórica: es el caso de ruido en el contexto de B.1. Los dos ceros de Answer
Relevancy (C2 e I3) no son una falla de recuperación: en C2 el mismo caso había dado 0.65 en la
corrida histórica con un pipeline idéntico, lo que muestra la variabilidad del juez.

**Abstención (criterio de igualdad exacta con la frase de escape).** E1 se abstiene
correctamente. E2 no: el básico agrega una explicación a la frase. Resultado: 14 de 15 casos con
abstención correcta.

### C.3. Comparación: Básico vs. Avanzado (Quince Casos)

| ID | Faithfulness (B / A) | Answer Relevancy (B / A) | Context Precision (B / A) | Context Recall (B / A) |
|---|---|---|---|---|
| S1 | 1.00 / 1.00 | 0.82 / 0.90 | 1.00 / 1.00 | 1.00 / 1.00 |
| S2 | 1.00 / 1.00 | 0.93 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| S3 | 1.00 / 1.00 | 0.77 / 0.76 | 1.00 / 1.00 | 1.00 / 1.00 |
| S4 | 1.00 / 1.00 | 0.88 / 0.94 | 1.00 / 1.00 | 1.00 / 1.00 |
| S5 | 1.00 / 1.00 | 0.94 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 |
| C1 | 0.80 / 0.50 | 0.74 / 0.67 | 0.00 / 1.00 | 0.67 / 0.67 |
| C2 | 1.00 / 1.00 | 0.00 / 0.68 | 0.50 / 0.50 | 1.00 / 1.00 |
| C3 | 0.78 / 0.78 | 0.82 / 0.81 | 1.00 / 1.00 | 1.00 / 1.00 |
| C4 | 0.71 / 0.90 | 0.81 / 0.74 | 1.00 / 1.00 | 1.00 / 1.00 |
| C5 | 1.00 / 0.60 | 0.65 / 0.68 | 0.50 / 0.83 | 1.00 / 1.00 |
| I1 | 1.00 / 1.00 | 0.59 / 0.68 | 1.00 / 1.00 | 1.00 / 1.00 |
| I2 | 1.00 / 1.00 | 0.76 / 0.83 | 1.00 / 1.00 | 1.00 / 1.00 |
| I3 | 1.00 / 1.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 |

| Métrica (n=13) | Básico | Avanzado | Delta |
|---|---|---|---|
| Faithfulness | 0.946 | 0.906 | −0.040 |
| Answer Relevancy | 0.670 | 0.665 | −0.005 |
| Context Precision | 0.846 | 0.949 | **+0.103** |
| Context Recall | 0.974 | 0.974 | 0.000 |

**La mejora está en Context Precision (+0.103).** Es el efecto que busca el pipeline
avanzado: el reranker deja pasar menos ruido. En C1 el avanzado incorpora `POL-CAM-001_chk1`
y la precisión pasa de 0.00 a 1.00; en C5 sube de 0.50 a 0.83. No se aísla el aporte del
chunking, de la recuperación amplia ni del reranker, porque no se evaluaron por separado.

**Answer Relevancy queda en empate técnico (−0.005).** La diferencia es menor que la
variabilidad del juez (ver C.4).

**Faithfulness baja 0.040, y no es uniforme.** Pesan dos casos, parcialmente compensados por
un tercero:
* **C1 (0.80 → 0.50).** El prompt nuevo cumple lo que se buscaba: pide la fecha de recepción
  y ya no afirma que la devolución venció. Pero la respuesta omite la gestión de la diferencia
  de precio que el cliente consultó y mantiene el mismo Recall (0.67) que el básico, así que
  C1 no queda resuelto del todo.
* **C5 (1.00 → 0.60).** El reranker no seleccionó `PROD-003`, la ficha del producto consultado
  (contexto `FAQ-003_chk2`, `FAQ-003_chk1`, `POL-CAM-003_chk2`), mientras que el básico sí la
  recuperó. Es un hecho observable; la causa del puntaje del juez no se puede confirmar con el
  JSON, que no conserva su explicación.
* **C4 (0.71 → 0.90)** compensa en parte la caída.

**Context Recall no se mueve (0.974 en ambos):** los dos recuperan la información clave,
salvo la que se pierde en C1 en ambos pipelines.

**Abstención exacta.** E1: ambos pipelines devuelven la frase exacta. E2: **ninguno cumple**.
El avanzado responde la frase de escape con un punto final, a pesar de la regla "sin
puntuación"; el básico agrega una explicación. Semánticamente el avanzado se abstiene
correctamente, pero el criterio de igualdad exacta lo rechaza. Resultado: 14 de 15 en ambos.

### C.4. Diagnóstico de la Métrica más Baja

La métrica más baja del RAG avanzado es **Answer Relevancy: 0.665 (n=13)**. En la corrida
histórica de diez casos era 0.645 (n=8), también la más baja.

**Diagnóstico previo.** En la corrida histórica la caída se concentraba en S2 (0.93 → 0.00) y
C3 (0.88 → 0.68). La hipótesis era que el prompt anterior obligaba a sumar condiciones y
evaluaciones documentadas además de lo preguntado, lo que favorecía información lateral
(I1 terminaba hablando de falta de información sobre el catálogo; S3 agregaba validaciones a
una pregunta sobre qué datos entregar).

**Acción aplicada.** El generador avanzado usa `REGLAS_GENERADOR_AVANZADO`, propio de
`parte_b/rag_avanzado.py`. Conserva el respaldo documental, la corrección de premisas falsas,
la protección ante instrucciones en los datos y los límites sobre operaciones, stock y estados
reales. La Parte A y el prompt del reranker permanecen intactos. El nuevo alcance exige:

1. Responder primero la pregunta concreta y todos sus aspectos respaldados.
2. Incluir solo condiciones indispensables, evitando información lateral no solicitada.
3. Distinguir compra de recepción; si falta la fecha de recepción, explicar los plazos
   condicionalmente y pedirla.
4. Responder lo respaldado de una consulta parcial y señalar el dato concreto faltante, sin
   usar la frase de escape como descargo parcial.
5. Si ninguna parte puede responderse, devolver únicamente la frase de escape exacta, sin
   puntuación ni explicaciones adicionales.

**Medición posterior.** Se volvió a correr RAGAS sobre ambos pipelines con los quince casos.
Para comparar antes y después sin mezclar casos, se usan los mismos 8 casos puntuados de la
corrida histórica. El básico funciona como control: su código y su prompt no cambiaron.

| Pipeline (mismos 8 casos) | Métrica | Histórica | Nueva | Diferencia |
|---|---|---|---|---|
| Avanzado | Answer Relevancy | 0.645 | 0.666 | +0.020 |
| Avanzado | Faithfulness | 0.890 | 0.910 | +0.020 |
| Avanzado | Context Precision | 0.937 | 0.937 | 0.000 |
| Avanzado | Context Recall | 0.958 | 0.958 | 0.000 |
| Básico (control) | Answer Relevancy | 0.768 | 0.678 | **−0.090** |
| Básico (control) | Faithfulness | 0.900 | 0.947 | +0.047 |

**Qué se puede y qué no se puede concluir.**
* Faithfulness del avanzado no bajó en los casos comparables (+0.020), que era la condición
  para considerar aceptable el cambio.
* Answer Relevancy del avanzado subió 0.020, pero el **control cambió −0.090 sin que se tocara
  nada**. Con un juez LLM no determinista y `strictness=1`, una diferencia de 0.020 no se puede
  atribuir al prompt. La mejora cuantitativa **no está demostrada**.
* Sí hay un cambio de comportamiento verificable: C1 ahora pide la fecha de recepción en lugar
  de concluir que el plazo venció.
* El objetivo principal no se logró: S2 sigue en 0.00 aunque la respuesta nueva define
  "despachado" de forma directa. Cierra con *"No cuento con información sobre la ubicación
  actual de tu paquete ni una fecha confirmada de entrega"*. I3 queda en 0.00 en **ambos**
  pipelines, y su respuesta correcta consiste justamente en declarar que falta ese dato. La
  hipótesis es que Answer Relevancy penaliza las respuestas que declaran falta de información,
  aunque sean correctas. El JSON no conserva la explicación del juez, así que es una hipótesis.
* La regla "sin puntuación" para la frase de escape no se cumplió en E2.

**Acciones técnicas propuestas (a futuro).**
1. **Normalizar la frase de escape en código, no en el prompt.** Si la respuesta del modelo,
   sin espacios ni puntuación final, equivale a la frase de escape, devolver exactamente
   `FRASE_ESCAPE`. Es coherente con el principio del TP (el LLM interpreta, el código decide) y
   corrige el caso E2 sin depender de que el modelo obedezca.
2. **Acotar el descargo final.** Ajustar el prompt para que, cuando ya se respondió lo
   preguntado, no cierre con una frase del tipo "no cuento con información" salvo que sea el
   dato que el cliente pidió. Apunta a S2 e I3, los dos ceros de Answer Relevancy.
3. **Reducir la varianza de la medición.** Repetir la evaluación al menos tres veces y
   reportar promedio y dispersión, o subir `strictness` de Answer Relevancy, antes de
   atribuir una diferencia de pocas centésimas a un cambio de prompt.
4. **Cobertura de C1.** Agregar al prompt o a la recuperación la gestión de la diferencia de
   precio (`POL-CAM-004`), omitida en la respuesta avanzada.

**Limitaciones.** Cada promedio sale de 13 preguntas puntuadas y de un único juez LLM. El
básico y el avanzado usan prompts distintos, de modo que el delta mezcla prompt y
recuperación. La referencia de C1 aplica los plazos como si los quince días fueran desde la
recepción, aunque la pregunta solo menciona la compra: se conservó por el requisito de no
modificar los diez casos originales, y una respuesta condicional que pide la fecha de
recepción puede ser correcta aunque difiera de esa referencia. Las bases corresponden a la
carga original, sin evento en caliente de la Entrega 2.

---

## Anexo C — Corrida histórica (diez casos, prompt anterior)

Se conservan las tablas de la corrida original, guardada en
[`resultados_ragas.json`](resultados_ragas.json), que no se modificó. Corresponden a diez
preguntas (n=8 puntuadas) y al prompt generador anterior, y no deben compararse directamente
con los promedios de quince casos.

### Anexo C.A. Línea Base Histórica (RAG Básico, Diez Casos)

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

### Anexo C.B. Comparación Histórica: Básico vs. Avanzado (Diez Casos)

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
