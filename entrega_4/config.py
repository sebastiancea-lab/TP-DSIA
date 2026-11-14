"""Entrega 4: configuración central del agente Portalia.

Importar este módulo ANTES que langchain/langgraph: ``load_dotenv`` tiene que correr
primero para que las variables ``LANGCHAIN_*`` de LangSmith se vean al activar el tracing.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
# Permite importar entrega_1/2/3 aunque se arranque desde otro directorio.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

load_dotenv(RAIZ / ".env")

DIRECTORIO_E4 = Path(__file__).resolve().parent
DIRECTORIO_DATOS = DIRECTORIO_E4 / "datos"
ARCHIVO_BACKEND = DIRECTORIO_DATOS / "backend_portalia.json"

MODELO_POR_DEFECTO = "gemini-3.1-flash-lite"  # el modelo con el que se validó la matriz

_PLACEHOLDER = re.compile(r"^your_\w*_here$", re.IGNORECASE)


def limpiar_valor(crudo: str) -> str:
    """Devuelve '' si el valor está vacío o es un placeholder tipo ``your_xxx_here``."""
    valor = (crudo or "").strip()
    return "" if _PLACEHOLDER.match(valor) else valor


def resolver_ruta_db(crudo: str) -> Path:
    """Ruta de la memoria SQLite. Una ruta relativa se toma desde la raíz del repo."""
    valor = (crudo or "").strip()
    if not valor:
        return DIRECTORIO_DATOS / "memoria.sqlite3"
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def _entero(nombre: str, defecto: int) -> int:
    crudo = os.environ.get(nombre, "").strip()
    try:
        valor = int(crudo) if crudo else defecto
    except ValueError:
        return defecto
    return valor if valor > 0 else defecto


LLM_API_KEY = limpiar_valor(os.environ.get("LLM_API_KEY", ""))
LLM_MODEL = limpiar_valor(os.environ.get("LLM_MODEL", "")) or MODELO_POR_DEFECTO

# Memoria persistente por thread_id (SqliteSaver). Se puede mover con DB_PATH.
DB_PATH = resolver_ruta_db(os.environ.get("DB_PATH", ""))

# Los tickets viven junto a la memoria (en Docker, en el mismo volumen persistente).
DB_SOLICITUDES = DB_PATH.parent / "solicitudes.sqlite3"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "").strip().upper() or "INFO"

# --- Agente ---------------------------------------------------------------
MAX_ITERACIONES = _entero("MAX_ITERACIONES", 5)   # vueltas agente <-> tools por turno
VENTANA_MENSAJES = 10                              # mensajes que ve el LLM
K_RETRIEVER = 5                                    # fragmentos de política por búsqueda
TEMPERATURA = 0
TAGS_TRAZA = ["entrega4"]

# --- Reglas de negocio (POL-CAM-001 / POL-DEV-001) ------------------------
PLAZO_CAMBIO_DIAS = 30
PLAZO_DEVOLUCION_DIAS = 10

# --- Guardrails -----------------------------------------------------------
MAX_LARGO_INPUT = 1000
