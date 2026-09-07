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
| `LLM_API_KEY` | API key de modelo LLM. **Obligatoria.** |
| `LLM_MODEL` | Modelo de LLM. Opcional; por defecto `gemini-3.6-flash`. |

`.env` está en `.gitignore` y no se sube al repositorio.
