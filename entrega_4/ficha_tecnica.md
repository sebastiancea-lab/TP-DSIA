# Ficha técnica — Agente de postventa de Portalia

| | |
|---|---|
| **Sistema** | Agente conversacional de atención postventa (API REST) |
| **Versión** | Entrega 4 (API 1.0.0) |
| **Dominio** | Portalia, e-commerce de productos electrónicos |
| **Responsables** | Mariel Garcik, Federico Chiesa, Lucas Rodríguez Goñi, Lucas Markowicz |

## Propósito

Atender consultas de postventa: políticas de cambios, devoluciones, garantías y envíos; estado de
un pedido; y registrar solicitudes de cambio o devolución cuando el motor de reglas las habilita.
**No es un sistema de decisión autónoma**: la elegibilidad la calcula Python, no el modelo.
Fuera de postventa responde con una frase de escape y no inventa.

## Modelo

- **LLM:** Gemini vía `langchain-google-genai` (`ChatGoogleGenerativeAI`), `temperature=0`.
  Modelo configurable con `LLM_MODEL`; el usado en las pruebas fue `gemini-3.1-flash-lite`.
- **Embeddings:** `gemini-embedding-001`, 768 dimensiones (colección Chroma de la Entrega 2).
- **Orquestación:** LangGraph (grafo ReAct manual), `langgraph-checkpoint-sqlite`.
- **Versiones:** las fijadas en `uv.lock` / `requirements.txt` (p. ej. `langgraph 1.2.16`,
  `langchain-google-genai 4.4.1`, `fastapi 0.162.7`).

## Herramientas

| Herramienta | Tipo | Qué hace |
|---|---|---|
| `buscar_politicas_postventa` | consulta | Trae fragmentos de política del retriever de la Entrega 3 (sin llamar al LLM). |
| `consultar_pedido` | sensor | Estado, fecha, ítems y elegibilidad calculada de un pedido. |
| `consultar_stock` | sensor | Stock por producto y variante. |
| `solicitar_cambio_o_devolucion` | actuador (escribe) | Registra un ticket solo si el motor de reglas lo permite; si no, devuelve el motivo. Idempotente. |

## Datos

- **Base de conocimiento:** 38 documentos (políticas, envíos, FAQ, procedimientos y 20 productos),
  limpiada por el ETL de la Entrega 2, en ChromaDB local (no se versiona; se regenera).
- **Backend mock:** 8 pedidos y stock de 20 productos en `entrega_4/datos/backend_portalia.json`.
  Datos personales ficticios, a propósito, para demostrar el enmascarado.
- **Memoria:** historial por `thread_id` en SQLite (`DB_PATH`). Tickets en `solicitudes.sqlite3`.
  Ambos contienen texto de las conversaciones: tratarlos como datos de clientes.

## Guardrails

| Capa | Control | Resultado |
|---|---|---|
| Input | Longitud (vacío o más de 1000 caracteres) | Respuesta fija, el grafo no se invoca |
| Input | Prompt injection (patrones ES/EN) | Respuesta fija, el grafo no se invoca |
| Agente | System prompt: anti-injection, solo datos de herramientas, frase de escape | — |
| Agente | Tope de 5 iteraciones y nodo de cierre | Corte controlado |
| Actuador | Plazo, stock y pedido validados en Python | Rechazo con motivo |
| Output | Enmascarado de email, teléfono, DNI y tarjeta | También se corrige en la memoria |

## Métricas y observabilidad

Logs JSON, `GET /metricas` (consultas, errores, latencia promedio y p95, herramientas, bloqueos,
tokens) y trazas en LangSmith (proyecto `tp-portalia`, tag `entrega4`).
Resultados: matriz de 4 escenarios (5 casos) con trayectorias correctas; 36 tests automáticos OK.
Costo medido: ≈ US$ 0,001 por consulta (ver `informe.md`).

## Riesgos y mitigaciones

| Riesgo | Mitigación | Riesgo residual |
|---|---|---|
| Prompt injection | Validator de input, regla en el prompt, escritura validada por el motor de reglas | Paráfrasis o idiomas no cubiertos por las regex |
| Fuga de datos personales | Enmascarado de output (también en memoria); el prompt pide no mostrarlos | Nombres propios no se enmascaran |
| Alucinación (promesas o datos inventados) | Respuestas solo desde herramientas, frase de escape, prompt con regla explícita | El LLM puede redactar mal; se vio y se corrigió un caso |
| Cambio o devolución indebidos | Elegibilidad en Python; el actuador rechaza | Bajo |
| Bucles de herramientas | Tope de iteraciones y `recursion_limit` | Bajo |
| Caída del LLM o de Chroma | LLM o red: 502 y el turno se cierra en memoria; falla del RAG: la tool devuelve un error al agente (200); almacenamiento: 500; `/health` degradado si Chroma no abre | Sin reintentos automáticos |
| Acceso sin autenticar | — | **Alto en producción**: la API no autentica |

## Límites de uso

- Pensado para una instancia y tráfico bajo; SQLite y métricas en memoria.
- Solo español; solo postventa de Portalia.
- Backend de demostración, no conectado a sistemas reales.
- Plan gratuito de Gemini: ~15 req/min, no apto para producción.
- Un humano debe revisar los casos que el agente no resuelve (PROC-001).

## Costos

Gemini 3.1 Flash-Lite, tarifa estándar tomada de fuentes de terceros (confirmar en la página
oficial): US$ 0,25 / 1 M tokens de entrada y US$ 1,50 / 1 M de salida. Promedio medido ≈ US$ 0,001
por consulta (≈ US$ 30 por mes con 1.000 consultas por día). Detalle por escenario en `informe.md`.
