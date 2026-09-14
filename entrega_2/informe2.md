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