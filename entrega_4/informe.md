# Informe — Entrega 4: agente Portalia con LangGraph, API, observabilidad y guardrails

## 1. Qué se construyó

El RAG de la Entrega 3 pasó a ser un **agente ReAct** que decide solo qué herramienta usar,
con memoria SQLite por `thread_id`, expuesto como **API FastAPI** con logs JSON, métricas,
trazas en LangSmith y guardrails de input y de output.

Principio que se mantiene desde la Entrega 1: **la IA interpreta, el motor de reglas decide**.
La elegibilidad de un cambio (30 días, POL-CAM-001) o una devolución (10 días, POL-DEV-001) y
la disponibilidad de stock las calcula `backend.py` en Python. El LLM nunca aprueba nada:
la herramienta actuadora rechaza con motivo si el motor de reglas dice que no.

## 2. Arquitectura

```
cliente ─► POST /consultar ─► guardrail de input ──(bloquea)──► respuesta fija, sin invocar el grafo
                                   │ pasa
                                   ▼
                     ┌────────────────────────────── grafo LangGraph (checkpointer SQLite) ─────┐
                     │  START ─► agente ──(tool_calls)──► tools ──┐                              │
                     │             ▲   │                          │                              │
                     │             └───┼──────────────────────────┘                              │
                     │                 ├─(sin tool_calls)──► END                                 │
                     │                 └─(iteraciones ≥ MAX)─► cierre ─► END                     │
                     └───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                     guardrail de output (enmascara PII, también en la memoria) ─► respuesta
```

| Módulo | Rol |
|---|---|
| `config.py` | Constantes y `load_dotenv` antes de importar langchain (para el tracing). |
| `prompts.py` | `SYSTEM_PROMPT` (reglas, anti-injection, frase de escape) y mensaje de corte. |
| `backend.py` + `datos/backend_portalia.json` | Pedidos y stock mock, elegibilidad, tickets en SQLite (idempotentes por pedido, tipo y producto). |
| `tools.py` | 4 herramientas: `buscar_politicas_postventa` (retriever de la E3), `consultar_pedido` y `consultar_stock` (sensores), `solicitar_cambio_o_devolucion` (actuador). |
| `agente.py` | Grafo manual, tope de iteraciones, nodo `cierre`, ventana de mensajes. |
| `memoria.py` | `SqliteSaver` en archivo (`DB_PATH`). |
| `guardrails.py` | Validators deterministas de input y output. |
| `observabilidad.py` | Logger JSON y `Metricas` con lock. |
| `api.py`, `schemas.py` | FastAPI: `/health`, `/consultar`, `/estado/{thread_id}`, `/metricas`. |

### Decisiones

- **Grafo manual y no `create_agent`:** el límite de iteraciones queda explícito. El estado lleva
  `iteraciones`; al llegar a `MAX_ITERACIONES` (5) el nodo `cierre` responde con un corte
  controlado. Como el último mensaje del agente tiene `tool_calls` sin respuesta, `cierre` agrega
  `ToolMessage` de cancelación: sin eso el historial quedaría inválido para el siguiente turno.
  `recursion_limit` queda como red de seguridad (se atrapa `GraphRecursionError`).
- **Contador por turno:** el estado persiste en el checkpointer, así que cada turno arranca con
  `iteraciones: 0` (`entrada_turno`).
- **Ventana de 10 mensajes** (`trim_messages`, `start_on="human"`): el LLM ve poco, el checkpointer
  guarda todo. Si un turno largo desplazara al mensaje humano, se usa el turno completo.
- **Un solo dueño de la memoria:** el grafo compilado lleva el checkpointer; el saver y el grafo
  se crean una vez en el `lifespan`.
- **Guardrails deterministas** (regex y Luhn para tarjetas): sin costo de tokens y testeables.
  Cuando el output se enmascara, el mensaje también se reemplaza en la memoria (`update_state`
  con el mismo `id`), para que la PII no vuelva a salir por `/estado`.
- **Errores por capas:** errores del proveedor del LLM o de red → 502; `sqlite3.Error` (memoria o
  tickets) → 500 con su propio mensaje; validación → 422. Todos suman en `errores` de `/metricas`.
  Una falla del RAG (`ErrorRAG`, o un 429 al embeber la consulta) **no** es un 502: la tool
  `buscar_politicas_postventa` la atrapa y se la devuelve al agente como
  `{"error", "errores"}`, así que la respuesta al cliente es 200. Si Chroma no abre, el servidor
  sigue arriba y `/health` informa `degradado`.
- **Turnos fallidos con la memoria coherente:** cuando un turno termina en 502 o en corte por
  recursión, la API lo cierra en el checkpointer (`update_state` con `mensajes_de_cierre`): sin
  mensaje humano huérfano ni `tool_calls` abiertos. Una respuesta vacía del LLM se reemplaza por
  la frase de escape, también en la memoria.
- **Fechas relativas en el mock:** `entregado_hace_dias` se calcula contra hoy, así los casos de
  elegibilidad de la demo no caducan. En los tests la fecha de referencia se inyecta.
- **Tickets junto a la memoria:** `solicitudes.sqlite3` vive en la carpeta de `DB_PATH` (en Docker,
  el mismo volumen).

## 3. Resultados

### Matriz de validación (Gemini real, `validacion_agente.py`)

Última corrida, modelo `gemini-3.1-flash-lite`, exit code 0:

| # | Escenario | Trayectoria observada | tool_selection | sin_prohibidas | contiene |
|---|---|---|---|---|---|
| 1 | Política de devolución | `buscar_politicas_postventa` | OK | OK | OK |
| 2 | Estado del pedido 48327 | `consultar_pedido` | OK | OK | OK |
| 3 | Cambio elegible (pedido 51204) | `consultar_pedido` → `consultar_stock` → `solicitar_cambio_o_devolucion` | OK | OK | OK |
| 4a | Devolución vencida (pedido 50718) | `consultar_pedido` (sin actuador) | OK | OK | OK |
| 4b | Fuera de dominio (Mundial 2022) | sin herramientas, frase de escape | OK | OK | OK |

En una corrida previa, el escenario 3 prometía "un mail con los pasos a seguir", algo que ninguna
herramienta devuelve. Se agregó la regla 7 al prompt (informar solo lo que devuelven las
herramientas) y en la corrida siguiente desapareció. Es una verificación puntual, no una garantía:
el prompt reduce el riesgo, no lo elimina.

### Tests automáticos (sin red ni API key)

`uv run python -m unittest entrega_4.security_tests entrega_4.test_agente_api -v` → **36 tests OK**.

- `security_tests.py` (12): injection bloqueada (11 variantes, incluidas las evasiones comunes),
  input largo y vacío, consultas legítimas sin falsos positivos (incluye "mostrame las
  instrucciones para devolver"), PII enmascarada, formatos de teléfono, texto sin PII intacto
  (códigos de producto, años, montos), DNI sin contexto de monto que sí se enmascara,
  configuración segura con un `.env` recién copiado (placeholders, `DB_PATH` relativo), y
  `/consultar` con injection → 200, `bloqueado=true`, **el LLM falso con 0 llamadas**.
- `test_agente_api.py` (24): la memoria crece entre turnos, **el thread sobrevive a un reinicio**
  (app nueva, saver nuevo, mismo archivo), threads aislados, `thread_id` generado, 404 y 422,
  uso de tools y métricas, el actuador respeta el motor de reglas, límite de iteraciones con
  historial válido después (y `recursion_limit` que sigue al `max_iteraciones` inyectado), PII
  enmascarada también en memoria, 502 ante error del LLM con el turno cerrado en memoria, corte
  por recursión guardado en memoria, respuesta vacía, 500 ante `sqlite3.Error`, falla del
  retriever devuelta al LLM como JSON de error, reglas del backend (misma variante, pedido
  cancelado, idempotencia con stock agotado, carrera en el `INSERT`), `/health` y `/metricas`.
- Los dos módulos fijan `LANGCHAIN_TRACING_V2=false` antes de importar `entrega_4`: aunque el
  `.env` del desarrollador tenga el tracing prendido, los tests con LLM falso no mandan trazas.
- Regresión de la Entrega 3: `test_rag_pipeline.py` (29) y `test_rag_hardening.py` (39) siguen OK.

### Prueba manual con el servidor real

Con `uvicorn` y Gemini: dos turnos con el mismo `thread_id` (el agente recordó el nombre y el
pedido), mensaje de injection bloqueado con `bloqueado=true`, servidor reiniciado y `/estado`
con los 4 mensajes intactos, 404 para un thread inexistente y `/metricas` coherente. Las métricas
**se reinician con el servidor** (viven en memoria); la conversación no.

## 4. Análisis de costos

Precio usado: **US$ 0,25 por millón de tokens de entrada y US$ 1,50 por millón de salida** para
`gemini-3.1-flash-lite` (tarifa estándar). Ojo: lo tomé de sitios que citan la tarifa de Google,
no de la página oficial (https://ai.google.dev/gemini-api/docs/pricing); hay que confirmarlo antes
de presentar. Los tokens salen de `usage_metadata` (suma de todas las llamadas al LLM del turno).

| Escenario | Llamadas al LLM | Tokens entrada | Tokens salida | Costo (US$) |
|---|---|---|---|---|
| 1 Política | 2 | 3.362 | 66 | 0,00094 |
| 2 Pedido | 2 | 3.034 | 60 | 0,00085 |
| 3 Cambio elegible | 4 | 6.785 | 175 | 0,00196 |
| 4a Devolución vencida | 2 | 3.068 | 96 | 0,00091 |
| 4b Fuera de dominio | 1 | 1.358 | 14 | 0,00036 |
| **Promedio** | | | | **≈ 0,0010** |

- **Costo por consulta:** alrededor de **US$ 0,001**. Proyección: 1.000 consultas por día ≈ US$ 1
  por día ≈ US$ 30 por mes; 10.000 por día ≈ US$ 300 por mes (suponiendo la misma mezcla).
- **Qué consume más:** el escenario 3 (cambio), por tener 4 llamadas al LLM con contexto creciente.
  En todos los casos domina la **entrada** (≥ 95 % de los tokens): el system prompt, el esquema
  de las 4 tools y los resultados de las herramientas se reenvían en cada vuelta. La tool de
  políticas trae hasta 5 fragmentos, así que es la que más pesa por llamada.
- **Palancas, si hiciera falta bajar el costo:** reducir `K_RETRIEVER`, acortar los docstrings de
  las tools y el prompt, o usar prompt caching. Y una barata: los bloqueos de input no gastan
  tokens (el grafo no se invoca).
- **Cuota:** con el plan gratuito, 15 req/min alcanzan para la matriz, pero un uso real
  necesita plan pago.

## 5. Observabilidad

- **Logs JSON** (una línea por evento): `ts`, `nivel`, `evento`, `thread_id`, `latencia_ms`,
  `herramientas`, `bloqueo`. No se loguea el texto del cliente.
- **`GET /metricas`:** consultas totales, errores, latencia promedio y p95, uso por herramienta,
  bloqueos por tipo (`input_injection`, `input_longitud`, `output_pii`) y tokens de entrada y salida.
- **LangSmith:** cada invoke lleva `metadata={"thread_id": ...}` y `tags=["entrega4"]`; proyecto
  `tp-portalia`. **Pendiente del equipo:** capturar las trazas (matriz y una sesión de demo) y
  guardarlas como `entrega_4/langsmith_*.png`; requiere la cuenta de LangSmith.

## 6. Limitaciones

- **Backend mock:** ocho pedidos y stock en JSON; no hay integración con un sistema real.
  Los datos personales son ficticios y están a propósito para demostrar el enmascarado.
- **Guardrails por patrones:** el detector de injection es por regex (español e inglés) y se puede
  evadir con paráfrasis o con otro idioma. Es una primera barrera, no la única: el motor de reglas
  y el actuador con validación siguen protegiendo la escritura aunque el LLM sea engañado.
  El enmascarado cubre email, teléfono, DNI y tarjeta; un nombre propio no se enmascara.
- **Métricas en memoria:** se pierden al reiniciar y no se agregan entre réplicas.
- **Sin autenticación** en la API ni confirmación fuerte de identidad del cliente: cualquiera que
  conozca un número de pedido puede consultarlo. Para producción hace falta autenticar.
- **SQLite:** alcanza para una instancia; con varias réplicas habría que pasar a otro checkpointer.
- **Docker:** los archivos viven en `entrega_4/` (contexto de build: la raíz del repo). Con
  `docker compose -f entrega_4/docker-compose.yaml up --build -d` el servicio quedó `(healthy)`,
  la imagen no contiene `.env` ni bases SQLite locales y `/consultar` respondió 200 desde el
  contenedor. Verificado en una máquina con Docker 28.3.0; no se probó en otro entorno.
- **Cuota free tier** de Gemini: las corridas reales son pocas a propósito.
