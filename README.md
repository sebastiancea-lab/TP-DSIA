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
- Sebastián Cea
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
| `LLM_MODEL` | Modelo de LLM. Opcional; por defecto `gemini-3.6-flash`. |
| `EMBEDDING_MODEL` | Modelo de embeddings (Entrega 2). Obligatoria para `entrega_2/`; por defecto `gemini-embedding-001`. |
| `EMBEDDING_DIMENSION` | Dimensión de los embeddings (Entrega 2). Obligatoria para `entrega_2/`; por defecto `768`. |

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
