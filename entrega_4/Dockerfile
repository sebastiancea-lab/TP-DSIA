# Alternativa opcional a "uv run": el camino principal del README es uv.
# Se construye con el contexto en la raíz del repo (ver entrega_4/docker-compose.yaml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    DB_PATH=/app/state/memoria.sqlite3

WORKDIR /app
RUN pip install --no-cache-dir uv

# Dependencias primero: la capa se reutiliza mientras no cambien pyproject/uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY entrega_1 ./entrega_1
COPY entrega_2 ./entrega_2
COPY entrega_3 ./entrega_3
COPY entrega_4 ./entrega_4

RUN mkdir -p /app/state

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "entrega_4.api:app", "--host", "0.0.0.0", "--port", "8000"]
