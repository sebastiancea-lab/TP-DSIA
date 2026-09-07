# C.3 — Lote de prueba y tabla de resultados

Se armaron 6 inputs del dominio a partir de la Matriz de Intenciones de B.3, incluyendo
obligatoriamente uno ambiguo/incompleto (caso 4) y uno de prompt injection con lenguaje
hostil (caso 5). Se corrieron por el pipeline con parte_c/app.py contra la API de
Gemini (gemini-3.1-flash-lite).

| # | Input (resumido) | Salida del modelo | ¿Validó Pydantic? | Tipo de error si falló |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Pide estado del pedido 48327 (no llegó). | {"intencion":"consultar_pedido","pedido_id":"48327","producto":null,"variante_actual":null,"variante_solicitada":null,"motivo":null} | Sí | — |
| 2 | Cambio de talle S→L, remera del pedido 48327. | {"intencion":"solicitar_cambio","pedido_id":"48327","producto":"remera","variante_actual":null,"variante_solicitada":"L","motivo":"me queda chica"} | Sí | — |
| 3 | Devolución con reintegro, campera del pedido 51204, llegó manchada. | {"intencion":"solicitar_devolucion","pedido_id":"51204","producto":"campera","variante_actual":null,"variante_solicitada":null,"motivo":"me llego manchada"} | Sí | — |
| 4 | Quiere un cambio pero no dice pedido ni producto ni variante. | {"intencion":"solicitar_cambio","pedido_id":null,"producto":null,"variante_actual":null,"variante_solicitada":null,"motivo":null} | No | ValidationError · @model_validator: una operación de escritura (solicitar_cambio) sin pedido_id no pasa al backend. |
| 5 | Intenta anular el prompt, pedir el listado de la base y forzar confirmación de un reembolso. | {"intencion":"otro","pedido_id":null,"producto":null,"variante_actual":null,"variante_solicitada":null,"motivo":null} | Sí | — |
| 6 | Consulta por un "pedido N° 9" (fuera del rango de 4–6 dígitos). | {"intencion":"consultar_pedido","pedido_id":"9","producto":null,"variante_actual":null,"variante_solicitada":null,"motivo":null} | No | ValidationError · @field_validator de pedido_id: se esperan 4 a 6 dígitos. |

Para reproducir un caso: uv run python parte_c/app.py --canal <whatsapp|web> "<texto>".