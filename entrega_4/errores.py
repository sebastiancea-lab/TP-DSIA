"""Entrega 4: errores de servicios externos (LLM, embeddings, red), en un solo lugar.

Lo comparten ``api.py``, ``tools.py`` y ``validacion_agente.py``. Vive aparte para evitar
imports circulares (``agente`` importa ``tools``).
"""

from __future__ import annotations

import httpx
from google.genai import errors as genai_errors
from langchain_google_genai._common import GoogleGenerativeAIError

# Fallas del proveedor del LLM o de la red: la API responde 502 (falla un servicio externo).
ERRORES_EXTERNOS = (
    genai_errors.APIError, GoogleGenerativeAIError, httpx.HTTPError, ConnectionError, TimeoutError,
)
