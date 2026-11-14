# TP Integrador

**Asignatura:** Desarrollo de Sistemas de Inteligencia Artificial

## Dominio elegido: Portalia

Portalia es un e-commerce que vende y despacha productos propios. El foco de este
trabajo es la **atención post-venta**: el cliente escribe por chat o
mail ("mi pedido no llegó", "quiero cambiar el talle", "cancelen la compra") y hoy una
persona lee cada mensaje, lo interpreta y lo resuelve consultando los sistemas internos.

El sistema que diseñamos usa un LLM como **intérprete** del reclamo (detecta la intención
y extrae parámetros como el número de pedido) y deja que la **base de datos y las APIs
sean la autoridad** sobre el estado real del pedido, el stock y las políticas de cambio.

## Integrantes

- Mariel Garcik
- Federico Chiesa
- Lucas Rodríguez Goñi
- Lucas Markowicz

## Cómo correr el script

Gestión de dependencias con **uv**. uv instala Python 3.12 y las dependencias fijadas en `uv.lock`.

### 1. Instalar dependencias

```bash
uv sync
```

### 2. Configurar credenciales

```bash
cp .env.example .env
```

Y completar en `.env`:

| Variable | Descripción |
|---|---|
| `LLM_API_KEY` | API key de modelo de LLM. **Obligatoria.** |
| `LLM_MODEL` | Modelo de LLM. Opcional; por defecto `gemini-3.1-flash-lite` (el validado en la Entrega 4). |
| `EMBEDDING_MODEL` | Modelo de embeddings (Entrega 2). Obligatoria para `entrega_2/`; por defecto `gemini-embedding-001`. |
| `EMBEDDING_DIMENSION` | Dimensión de los embeddings (Entrega 2). Obligatoria para `entrega_2/`; por defecto `768`. |
| `DB_PATH` | Archivo SQLite de la memoria del agente (Entrega 4). Opcional; por defecto `entrega_4/datos/memoria.sqlite3`. |
| `MAX_ITERACIONES` | Tope de vueltas agente-herramientas por consulta (Entrega 4). Opcional; por defecto `5`. |
| `LOG_LEVEL` | Nivel de los logs JSON (Entrega 4). Opcional; por defecto `INFO`. |

`.env` está en `.gitignore` y no se sube al repositorio.

3. Ejecutar el pipeline (Parte C):

   ```bash
   # C.1 — smoke tests del contrato Pydantic, sin API key
   uv run python parte_c/schemas.py

   # C.2 — procesa uno o varios inputs (sin texto usa el caso guía del dominio)
   uv run python parte_c/app.py
   uv run python parte_c/app.py "mi pedido 48327 no llegó, ¿dónde está?"
   uv run python parte_c/app.py --canal web "quiero devolver la campera del pedido 51204"
   uv run python parte_c/app.py "texto 1" "texto 2" "texto 3"
   ```

   Canales admitidos por el contrato B.5a: `whatsapp` (default) y `web`. La tabla
   de C.3 en `resultados_lote.md` se completa con la salida de estos comandos
   sobre los 6 inputs de prueba.

## Entrega 2 — Base de Conocimiento Vectorial

La Entrega 2 usa las mismas credenciales de la tabla de arriba, más `EMBEDDING_MODEL` y
`EMBEDDING_DIMENSION` (ya están en `.env.example` con sus valores por defecto). Todos los
scripts de `entrega_2/` usan paths absolutos derivados de su propia ubicación, así que corren
igual desde la raíz del repo o desde `entrega_2/`.

El orden de ejecución importa: `etl_purga.py` va **primero**, porque genera
`base_conocimiento_limpia.json`, que es la entrada real de A.4 (FAISS) y B.1 (ChromaDB).
`base_conocimiento.json` es la base cruda —incluye a propósito 3 casi-duplicados y 2
inconsistencias estructurales para poder probar el ETL— y nunca se consume directamente.

```bash
uv sync
cp .env.example .env    # completar las variables de la tabla de arriba

# B.5 — ETL y purga semantica: genera base_conocimiento_limpia.json
uv run python entrega_2/etl_purga.py

# A.2 — similitud coseno a mano (sin API key)
uv run python entrega_2/a2_validacion.py

# A.4 — construye o recarga el indice FAISS y corre 3 consultas de prueba
uv run python entrega_2/pipeline_vectorial.py

# A.5 — prueba de volatilidad de la RAM (4 procesos independientes, en este orden)
uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py ram
uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py verificar-ram
uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py guardar
uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py recargar

# B.1 — migra la base limpia a ChromaDB (PersistentClient, coleccion coseno)
uv run python entrega_2/vector_db.py --cargar

# B.3 — evento de negocio en caliente (extiende el plazo de cambios a 45 dias)
uv run python entrega_2/vector_db.py --evento-caliente

# B.4 — busqueda hibrida (semantica + filtro nativo por metadata)
uv run python entrega_2/vector_db.py "me queda chica la remera y quiero otro talle" --categoria cambios
```

`entrega_2/indices/` (FAISS) y `entrega_2/chroma_db/` (ChromaDB) están en `.gitignore`: se
regeneran localmente y nunca se commitean. Los resultados de las Killer Queries (B.6) están en
`entrega_2/resultados_killer_queries.md`, y el informe completo (Partes A, B y C) en
`entrega_2/informe_entrega2.md`.

## Entrega 3 — Sistema RAG con LangChain LCEL y RAG Avanzado

Las dependencias de todas las entregas se gestionan en `pyproject.toml` y `uv.lock`. `requirements.txt` es una exportación de compatibilidad.

### Ejecución del Pipeline Unificado (Partes A y B)

Desde la raíz del repositorio, usando el entorno virtual `.venv`:

```bash
# RAG Básico (Parte A) — k=3 sobre colección persistente original
uv run python entrega_3/rag_pipeline.py --modo basico --k 3 "¿Hasta cuándo puedo solicitar una devolución?"

# RAG Avanzado (Parte B) — k=8 candidatos, chunking con solapamiento y LLM Reranker
uv run python entrega_3/rag_pipeline.py --modo avanzado "Compré unos auriculares inalámbricos hace 15 días, no me gusta el sonido y quiero saber si puedo cambiarlos por otro modelo de diferente precio o pedir que me devuelvan la plata"

# Reconstruye la colección avanzada (chunking con solapamiento)
uv run python entrega_3/rag_pipeline.py --reindexar
```

`entrega_3/parte_b/chroma_db/` (colección avanzada) también está en `.gitignore`.
Las consultas abren y validan la persistencia existente: si falta la base, la colección
está vacía o su configuración es incompatible, informan el problema sin reconstruirla.
La creación o reconstrucción avanzada se ejecuta únicamente con `--reindexar`, consume
embeddings y reemplaza la colección avanzada existente. El modo básico utiliza la colección
original de Entrega 2; el avanzado mantiene su propia colección.

### Evaluación con RAGAS (Parte C)

```bash
# Valida el golden dataset (15 preguntas, 5/5/2/3) sin llamar a la API
uv run python entrega_3/evaluacion_ragas.py --solo-dataset

# Evalúa ambos pipelines en un experimento nuevo, sin sobrescribir resultados históricos
uv run python entrega_3/evaluacion_ragas.py --experimento prueba-01

# Retoma esa corrida con la misma selección de pipelines
uv run python entrega_3/evaluacion_ragas.py --experimento prueba-01 --resume
```

La evaluación consume llamadas a Gemini y puede esperar ante errores 429.
Cada experimento guarda manifiesto, parciales y `resultados.json` en
`entrega_3/resultados_ragas/<experimento>/` (excluido por `.gitignore`). Si se omite
`--experimento`, se genera un identificador nuevo y se muestra en terminal.
`--resume` requiere ese identificador y rechaza cambios en dataset, código, modelos,
dependencias o documentos y metadatos persistidos, así como parciales inválidos y corridas
ya finalizadas. Para otra configuración, usar un identificador nuevo. El archivo histórico
`entrega_3/resultados_ragas.json` y los parciales anteriores no se sobrescriben ni se importan.
La abstención exacta y la coincidencia parcial con la frase de escape se registran por
separado; solo la frase exacta aprueba un caso de escape.

El dataset actual tiene quince casos y el generador avanzado usa un prompt propio;
sus cambios y limitaciones están en [`entrega_3/informe_entrega3.md`](entrega_3/informe_entrega3.md).
La evaluación vigente (quince preguntas y el prompt avanzado actual) está en
`entrega_3/resultados_ragas_15casos.json`, copia del `resultados.json` del experimento
`entrega3-15casos`. `entrega_3/resultados_ragas.json` conserva exclusivamente la corrida
histórica de diez preguntas y el prompt anterior; no deben compararse directamente.
La documentación técnica de la Parte A se encuentra en [`entrega_3/parte_a/README.md`](entrega_3/parte_a/README.md).

## Entrega 4 — Agente con LangGraph, API REST, observabilidad y guardrails

El agente de Portalia elige solo qué herramienta usar (políticas, pedido, stock, solicitud de
cambio o devolución), recuerda la conversación por `thread_id` en SQLite y se expone como API
FastAPI con logs JSON, `/metricas`, trazas en LangSmith y guardrails de input y output.
La elegibilidad de un cambio o una devolución la decide el motor de reglas en Python, no el LLM.
Detalle de diseño y resultados: [`entrega_4/informe.md`](entrega_4/informe.md) y
[`entrega_4/ficha_tecnica.md`](entrega_4/ficha_tecnica.md).

### Puesta en marcha (máquina limpia)

```bash
uv sync
cp .env.example .env          # completar solo LLM_API_KEY; para trazas: LANGCHAIN_API_KEY y LANGCHAIN_TRACING_V2=true

# La base Chroma no se versiona: hay que generarla (consume embeddings)
uv run python entrega_2/etl_purga.py
uv run python entrega_2/vector_db.py --cargar

uv run uvicorn entrega_4.api:app --port 8000      # docs interactivas en http://localhost:8000/docs
```

### Probar los endpoints

```bash
curl localhost:8000/health

curl -X POST localhost:8000/consultar -H "Content-Type: application/json"   -d '{"pregunta": "Donde esta mi pedido 48327?", "thread_id": "demo1"}'

curl localhost:8000/estado/demo1      # historial del thread (404 si no existe)
curl localhost:8000/metricas          # consultas, errores, latencia promedio y p95, herramientas, bloqueos
```

Si no se manda `thread_id`, la API genera uno y lo devuelve. En Windows, `curl` con tildes en
el cuerpo puede fallar por la codificación de la consola; usá PowerShell con
`Invoke-RestMethod` o el botón "Try it out" de `/docs`.

Pedidos de la demo (backend mock en `entrega_4/datos/backend_portalia.json`; los días de
entrega son relativos a hoy): `48327` despachado, `51204` entregado hace 3 días (cambio y
devolución elegibles), `50718` entregado hace 15 días (devolución vencida, cambio elegible),
`49880` entregado hace 45 días (nada elegible), `47990` con dos productos.

### Demo de memoria con reinicio

1. `POST /consultar` con `thread_id: "demo1"` ("Hola, soy Marcos, ¿dónde está el pedido 48327?").
2. Cortar el servidor (Ctrl+C) y volver a levantarlo con el mismo comando.
3. `GET /estado/demo1` devuelve el historial completo.
4. `POST /consultar` con el mismo `thread_id` ("¿qué pedido era?"): el agente lo recuerda.

Las métricas viven en memoria y se reinician con el servidor; la conversación no.

### Tests y validación

```bash
# Sin red ni API key (LLM falso): guardrails, API, memoria y reinicio
uv run python -m unittest entrega_4.security_tests entrega_4.test_agente_api -v

# Matriz de 4 escenarios contra Gemini real (trayectoria de herramientas); exit 1 si falla
uv run python entrega_4/validacion_agente.py
```

### Docker (alternativa opcional)

Requiere el `.env` y la base Chroma ya cargada en `entrega_2/chroma_db` (se monta como volumen).
Los archivos Docker viven en `entrega_4/`; el contexto de build es la raíz del repo, así que
los comandos se corren desde ahí:

```bash
docker compose -f entrega_4/docker-compose.yaml up --build -d
docker compose -f entrega_4/docker-compose.yaml ps     # el servicio debe quedar (healthy)
curl localhost:8000/health
```

La memoria queda en el volumen `memoria`; `docker compose -f entrega_4/docker-compose.yaml down -v`
la borra.

