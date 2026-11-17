# Plan: TP Final (Entrega 4): agente Portalia con LangGraph, API, observabilidad y guardrails

## Contexto

`entrega_4/requerimientos.md` pide convertir el RAG de la E3 en un agente LangGraph que elige sus propias herramientas y tiene memoria SQLite por `thread_id`. El agente se expone como API FastAPI, con logs, métricas y trazas en LangSmith, y lleva guardrails de input y de output. Se cierra con informe, ficha técnica y defensa.

Partimos de `fix/entrega-3-rag-hardening`, la rama más actualizada. Hoy esa rama **no tiene** nada de lo siguiente:
- un LLM con tool-calling (`GeminiChatModel` aplana los mensajes a texto),
- FastAPI,
- un checkpointer SQLite,
- un backend de pedidos o stock (no existe ningún dataset de pedidos).

Decisiones tomadas:
- **LLM:** `langchain-google-genai` (`ChatGoogleGenerativeAI`), alimentado con `LLM_API_KEY`.
- **Despliegue:** uv es el camino principal y Docker queda como alternativa opcional.
- **Ejecución:** secuencial, de punta a punta.

Reglas que valen durante todo el trabajo:
- Todo en español rioplatense.
- **Nada de `git add` ni `commit`**: los cambios quedan sin stagear para que los revise el equipo.
- Se mantiene "la IA interpreta, el motor de reglas decide": la elegibilidad de un cambio o una devolución la calcula Python, nunca el LLM.

## Estructura propuesta

```
entrega_4/
  config.py            # constantes + load_dotenv ANTES de importar langchain (tracing)
  prompts.py           # SYSTEM_PROMPT del agente (reglas, anti-injection, frase de escape)
  schemas.py           # Pydantic de la API (ConsultaRequest/Response, EstadoResponse, MetricasResponse)
  datos/backend_portalia.json   # mock: pedidos (con PII de cliente) + stock por PROD-xxx/variante
  backend.py           # motor de reglas: obtener_pedido, stock, elegibilidad, registrar_solicitud (SQLite)
  tools.py             # las 4 tools LangChain
  agente.py            # grafo ReAct manual + límite de iteraciones + ventana de mensajes
  memoria.py           # SqliteSaver persistente (DB_PATH)
  guardrails.py        # validators deterministas de input/output
  observabilidad.py    # logger JSON + Metricas (thread-safe)
  api.py               # FastAPI (lifespan, /health, /consultar, /estado/{thread_id}, /metricas)
  validacion_agente.py # matriz de 4 escenarios, evaluación de trayectoria, exit 1 si falla
  test_agente_api.py   # unittest con LLM falso: memoria por HTTP, reinicio, /estado, /metricas
  security_tests.py    # ≥4 tests de guardrails (unittest)
  informe.md, ficha_tecnica.md, fases_entrega_4.md
Dockerfile, docker-compose.yaml, .dockerignore   (raíz)
```

Todo se importa como `entrega_4.xxx` desde la raíz, igual que los tests de la E3. La API se levanta con `uv run uvicorn entrega_4.api:app --port 8000`.

> Pendiente para el equipo: la consigna no aclara si `security_tests.py`, `informe.md` y `ficha_tecnica.md` van en la raíz o en `entrega_4/`. Propongo `entrega_4/`, por consistencia con las entregas anteriores. Si el equipo prefiere la raíz, se mueven.

## Fases

### Fase 0: dependencias y configuración
- `uv add fastapi langgraph langgraph-checkpoint-sqlite langchain-google-genai langsmith uvicorn httpx`.
  - Ya se probó que resuelven sin conflicto con `langchain-core==1.6.9` y `langchain-community<0.4` (quedaron `fastapi 0.162.5`, `langchain-google-genai 4.4.1`, `langgraph-checkpoint-sqlite 3.1.3`). No tocar esos pins.
  - Si en algún momento `langchain-google-genai` choca con el pin, el plan B es agregarle `bind_tools` a `GeminiChatModel` (`entrega_3/parte_b/rag_avanzado.py:219`).
- Regenerar `requirements.txt` con `uv export --locked`, como se hace hoy.
- `.env.example`: agregar `DB_PATH`, `MAX_ITERACIONES` y `LOG_LEVEL`. Los `LANGCHAIN_*` se mantienen, porque ya está el proyecto `tp-portalia`.
- `.gitignore`: el patrón `*.sqlite3` ya cubre las bases nuevas. Usar esa extensión para la memoria y para los tickets.

### Fase 1: backend mock y motor de reglas (`backend.py` + `datos/backend_portalia.json`)
- **Datos:** unos 8 pedidos de 4–6 dígitos (incluido el 48327 del caso guía). Cada pedido tiene `estado`, `fecha_entrega`, `items` con `PROD-xxx`/variante y datos del cliente (nombre, email, teléfono, DNI). La PII está a propósito, para demostrar el enmascarado.
- **Stock:** por producto y variante, alineado con los `PROD-001..020` de `entrega_2/base_conocimiento_limpia.json`.
- **Elegibilidad:** cambio hasta 30 días desde la entrega (POL-CAM-001) y devolución hasta 10 días (POL-DEV-002), calculado en Python. Las ventanas son constantes con nombre y la fecha de referencia se puede inyectar, para que los tests sean deterministas.
- **`registrar_solicitud`** persiste tickets en SQLite (`entrega_4/datos/solicitudes.sqlite3`) y es idempotente por pedido y tipo.
- **`pedido_id`** se normaliza con el validator de `ExtraccionPostventa` (`entrega_1/parte_c/schemas.py:73`), sin duplicar la regex.

### Fase 2: herramientas (`tools.py`)
Cada docstring dice cuándo usar la tool y cuándo **no**, los argumentos cerrados son `Literal` y todas devuelven JSON. Ante un fallo devuelven `{"error": "...", "errores": [...]}`, nunca una excepción.

1. **`buscar_politicas_postventa(consulta)`**: es el retriever de la E3. Usa `crear_retriever(abrir_coleccion_existente(), k)` y `formatear_documentos` de `entrega_3/parte_a/rag_pipeline.py` (`:108`, `:139`, `:224`), construidos una sola vez. No llama al LLM: solo trae el contexto. Atrapa `ErrorRAG`.
2. **`consultar_pedido(pedido_id)`**: sensor. Devuelve el estado, los ítems, la fecha y la elegibilidad calculada.
3. **`consultar_stock(producto_id, variante)`**: segundo sensor (opcional pero útil para cambios).
4. **`solicitar_cambio_o_devolucion(pedido_id, tipo: Literal["cambio","devolucion"], motivo, variante_solicitada=None)`**: actuador. Solo escribe si `backend` dice que es elegible; si no, devuelve el motivo del rechazo.

### Fase 3: agente y memoria (`agente.py`, `memoria.py`, `prompts.py`)
- **Grafo manual:** `StateGraph` con nodo `agente` → condicional → `ToolNode(handle_tool_errors=True)` → `agente`. Se usa manual y no `create_agent` porque así el límite de iteraciones queda explícito y defendible.
- **Límite de iteraciones:** el estado lleva el campo `iteraciones`. Al llegar a `MAX_ITERACIONES` (5), el agente va a un nodo `cierre` que responde con un mensaje de corte controlado. Además se pasa `recursion_limit` como red de seguridad y se atrapa `GraphRecursionError`.
- **Fábrica:** `crear_agente(checkpointer, llm=None)` acepta un LLM inyectado, para correr los tests sin red. El LLM real es `ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=LLM_API_KEY, temperature=0)`.
- **Ventana:** el LLM ve solo los últimos ~10 mensajes (`trim_messages`, `start_on="human"`); el checkpointer guarda el historial completo.
- **Memoria:** `SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))`, siguiendo el patrón de `iads/10_clase/src/memory.py`.
- **System prompt:**
  - usar las tools para datos reales;
  - nunca confirmar un cambio o una devolución sin la tool actuadora;
  - responder con las políticas solo desde lo que devuelve el retriever;
  - frase de escape cuando no hay datos;
  - regla anti-injection (la regla 6 de `SYSTEM_PROMPT` de la E1).

### Fase 4: matriz de validación (`validacion_agente.py`)
Cuatro escenarios con `esperadas`, `prohibidas` y `contiene`, siguiendo el patrón de `iads/09_clase/.../03_eval_trayectoria.py`.

| # | Escenario | Trayectoria esperada |
|---|---|---|
| 1 | Consulta de política ("¿cuántos días tengo para devolver?") | `buscar_politicas_postventa`, sin el actuador |
| 2 | Estado de pedido ("¿dónde está mi pedido 48327?") | `consultar_pedido` |
| 3 | Cambio elegible (pedido dentro de 30 días, talle distinto) | `consultar_pedido` → (`consultar_stock`) → `solicitar_cambio_o_devolucion` |
| 4 | Fuera de dominio o inelegible (devolución vencida, o pregunta ajena) | sin el actuador; abstención o rechazo con motivo |

Cada caso corre con un `thread_id` nuevo. El script imprime una tabla y sale con exit code 1 si falla `tool_selection` o `sin_prohibidas`. Hace llamadas reales, pocas, por la cuota.

### Fase 5: guardrails y `security_tests.py`
**Validators de input** (deterministas, en Python):
- `validar_longitud`: corta arriba de `MAX_LARGO_INPUT` caracteres y rechaza los mensajes vacíos.
- `detectar_prompt_injection`: patrones regex en español e inglés ("ignorá las instrucciones", "system prompt", "actuá como", "olvida tus reglas", etc.).

**Validator de output:**
- `enmascarar_pii`: enmascara emails, teléfonos, DNI y tarjetas.

Los validators se integran en `/consultar`:
- Si el input se bloquea, se responde con un mensaje fijo, sin invocar el grafo, y cuenta como bloqueo.
- Si el output se enmascara, también se reemplaza el mensaje en la memoria con `update_state` (mismo `id`), siguiendo `iads/11_clase/.../api.py`.

**`security_tests.py`** (unittest, sin red), con al menos 6 tests:
- injection bloqueada;
- input largo bloqueado;
- input legítimo que pasa (sin falsos positivos);
- PII enmascarada;
- texto sin PII que queda intacto;
- `/consultar` vía `TestClient` con LLM falso: injection → 200 con `bloqueado=True` y el grafo sin invocar.

### Fase 6: observabilidad (`observabilidad.py`)
- **Logger JSON:** un formatter propio sobre `logging` emite una línea JSON por evento, con `ts`, `nivel`, `evento`, `thread_id`, `latencia_ms`, `herramientas` y `bloqueo`.
- **`Metricas`** (con `threading.Lock`) cuenta:
  - consultas totales,
  - errores,
  - latencias, de donde salen el promedio y el p95 (`statistics.quantiles` o percentil por orden),
  - uso de cada herramienta, extraído de `AIMessage.tool_calls` de los mensajes nuevos del turno,
  - bloqueos por tipo (input injection, input longitud, output PII).
- **LangSmith:** se mantienen las variables `LANGCHAIN_*`. Cada invoke lleva `metadata={"thread_id": ...}` y `tags=["entrega4"]`. Los tokens salen de `usage_metadata`, que `ChatGoogleGenerativeAI` reporta.

### Fase 7: API REST (`api.py`, `schemas.py`)
- **Lifespan:** construye una sola vez el saver, el grafo, el retriever y las métricas, y los guarda en `app.state`.
- **Endpoints**, todos `def` síncronos (corren en el threadpool):
  - `GET /health`: estado, modelo y si la colección Chroma está disponible.
  - `POST /consultar {pregunta, thread_id?}`: si no viene `thread_id`, genera un uuid. Pasa por el guardrail de input, después `graph.invoke`, después el de output. Responde `{thread_id, respuesta, herramientas_usadas, bloqueado, motivo_bloqueo, latencia_ms}`.
  - `GET /estado/{thread_id}`: devuelve el historial humano y asistente desde `get_state`, o 404 si el thread no existe.
  - `GET /metricas`.
- **Errores por capas:** `ErrorRAG` y los errores del LLM (`google.api_core` / `httpx`) dan 502, y los de validación dan 422. Nunca un `except` genérico que los mezcle.
- **`test_agente_api.py`** (unittest con un `FakeLLM` que extiende `BaseChatModel`, como en `iads/10_clase/tests`), con tmp_path para la DB:
  - la memoria crece entre turnos;
  - **el thread sobrevive a un reinicio**: una app nueva con un saver nuevo sobre el mismo archivo;
  - los threads no se mezclan;
  - `/estado` devuelve 404 para un thread inexistente;
  - `/metricas` cuenta bien.

### Fase 8: despliegue y README
- **Sección "Entrega 4" en el README raíz**, pensada para una máquina limpia:
  1. `uv sync`;
  2. `cp .env.example .env`;
  3. `etl_purga.py` y `vector_db.py --cargar` (la base Chroma no se versiona);
  4. `uvicorn`;
  5. curl de cada endpoint;
  6. demo de memoria: consultar, reiniciar el server, `/estado` y consultar de nuevo con el mismo `thread_id`;
  7. tests y matriz.
- **`Dockerfile` + `docker-compose.yaml`**: Python 3.12-slim con uv, `uv sync --frozen`, volumen `memoria` para el `DB_PATH`, `./entrega_2/chroma_db` montado, `env_file: .env` y healthcheck sobre `/health`. `.dockerignore` deja afuera `.env`.

### Fase 9: LangSmith, costos e informes
- Correr la matriz y una sesión de demo con tracing activo y capturar las trazas en `entrega_4/langsmith_*.png`.
- **Análisis de costos** en el `informe.md`: tokens de entrada y salida por consulta según el escenario, multiplicados por el precio del modelo Gemini. Hay que verificar el precio vigente al momento de escribir. Incluye costo por consulta, proyección a N consultas por día y qué tool o escenario consume más.
- **`informe.md`:** arquitectura (diagrama del grafo), decisiones, resultados de la matriz, de los tests de seguridad y de las métricas, y limitaciones (backend mock, cuota free tier).
- **`ficha_tecnica.md`:** propósito, modelo y versión, tools, datos, métricas, riesgos y mitigaciones, guardrails, límites de uso, costos y responsables.
- **`fases_entrega_4.md`:** seguimiento con checkboxes, igual que en la E3.

### Fase 10: preparar la defensa
Guion de 15 minutos: presentación (3), demo en vivo (7) y preguntas (5). La demo sigue esta secuencia:
1. consulta de política;
2. estado de un pedido, con la PII enmascarada;
3. cambio elegible;
4. intento de injection bloqueado;
5. reinicio del server y memoria intacta;
6. `/metricas` y la traza en LangSmith.

## Verificación
- `uv run python -m unittest entrega_4.security_tests entrega_4.test_agente_api -v`, sin red ni API key.
- Los tests existentes siguen verdes: `cd entrega_3/parte_a && uv run python test_rag_pipeline.py` y `uv run python -m unittest entrega_3.test_rag_hardening`.
- `uv run python entrega_4/validacion_agente.py` contra Gemini real: los 4 escenarios OK y exit 0.
- Manual:
  1. `uv run uvicorn entrega_4.api:app --port 8000`, después `curl` a `/health`, `/consultar` (dos turnos con el mismo `thread_id`) y `/estado/{id}`;
  2. matar y relevantar el server, y comprobar que `/estado` conserva el historial;
  3. `/metricas` refleja las consultas, el p95 y los bloqueos;
  4. la traza aparece en el proyecto `tp-portalia`.
- `docker compose up --build -d` y `/health` en `(healthy)`.
- Nada se stagea ni se commitea: se avisa al usuario que los cambios están listos para revisión.
