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

### Validación con NumPy

```python
import numpy as np

def similitud_coseno(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

q = np.array([8, 3])
doc_a = np.array([9, 2])
doc_b = np.array([2, 9])
doc_c = np.array([7, 5])

print("Q vs A:", similitud_coseno(q, doc_a))
print("Q vs B:", similitud_coseno(q, doc_b))
print("Q vs C:", similitud_coseno(q, doc_c))