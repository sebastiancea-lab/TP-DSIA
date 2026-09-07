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
- Lucas Marcowicz

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
