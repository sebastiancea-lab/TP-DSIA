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
# Permite importar los módulos de las entregas anteriores aunque el proceso
# arranque desde otro directorio.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Se carga .env al importar config, antes de LangChain/LangGraph, para que las
# variables de observabilidad de LangSmith estén disponibles al inicializarse.
load_dotenv(RAIZ / ".env")

DIRECTORIO_E4 = Path(__file__).resolve().parent
DIRECTORIO_DATOS = DIRECTORIO_E4 / "datos"
ARCHIVO_BACKEND = DIRECTORIO_DATOS / "backend_portalia.json"

MODELO_POR_DEFECTO = "gemini-3.1-flash-lite"  # el modelo con el que se validó la matriz

_PLACEHOLDER = re.compile(r"^your_\w*_here$", re.IGNORECASE)


def limpiar_valor(crudo: str) -> str:
    """Devuelve '' si el valor está vacío o es un placeholder tipo ``your_xxx_here``."""
    # Un valor de ejemplo no debe tratarse como una credencial válida.
    valor = (crudo or "").strip()
    return "" if _PLACEHOLDER.match(valor) else valor


def resolver_ruta_db(crudo: str) -> Path:
    """Ruta de la memoria SQLite. Una ruta relativa se toma desde la raíz del repo."""
    # Así DB_PATH funciona igual desde una terminal local y desde el contenedor,
    # aunque cada uno tenga distinto directorio de trabajo.
    valor = (crudo or "").strip()
    if not valor:
        return DIRECTORIO_DATOS / "memoria.sqlite3"
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def _entero(nombre: str, defecto: int) -> int:
    # Protege los límites configurables: entradas inválidas o no positivas usan
    # el valor seguro por defecto.
    crudo = os.environ.get(nombre, "").strip()
    try:
        valor = int(crudo) if crudo else defecto
    except ValueError:
        return defecto
    return valor if valor > 0 else defecto


LLM_API_KEY = limpiar_valor(os.environ.get("LLM_API_KEY", ""))
# Si no se especifica modelo, se usa el que se validó para esta entrega.
LLM_MODEL = limpiar_valor(os.environ.get("LLM_MODEL", "")) or MODELO_POR_DEFECTO

# La memoria conversacional se persiste por thread_id con SqliteSaver; DB_PATH
# permite ubicarla fuera de la ruta predeterminada.
DB_PATH = resolver_ruta_db(os.environ.get("DB_PATH", ""))

# Los tickets quedan junto a la memoria para compartir su persistencia, incluso
# dentro del volumen montado por Docker.
DB_SOLICITUDES = DB_PATH.parent / "solicitudes.sqlite3"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "").strip().upper() or "INFO"

# --- Parámetros del agente -----------------------------------------------
MAX_ITERACIONES = _entero("MAX_ITERACIONES", 5)   # vueltas agente <-> tools por turno
VENTANA_MENSAJES = 10                              # mensajes que ve el LLM
K_RETRIEVER = 5                                    # fragmentos de política por búsqueda
TEMPERATURA = 0
TAGS_TRAZA = ["entrega4"]

# --- Reglas de negocio (POL-CAM-001 / POL-DEV-001) ------------------------
# Plazos deterministas: el agente consulta estos valores; no debe inventarlos.
PLAZO_CAMBIO_DIAS = 30
PLAZO_DEVOLUCION_DIAS = 10

# --- Guardrails -----------------------------------------------------------
# Limita el tamaño del mensaje recibido antes de procesarlo.
MAX_LARGO_INPUT = 1000
