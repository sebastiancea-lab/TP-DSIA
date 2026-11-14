"""Entrega 4: memoria persistente por thread_id (SqliteSaver)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from entrega_4.config import DB_PATH


def crear_saver(ruta: Path | str = DB_PATH) -> SqliteSaver:
    """Saver SQLite en archivo: el historial sobrevive al reinicio del servidor.

    ``check_same_thread=False`` porque FastAPI atiende los endpoints sync en un
    threadpool; el saver serializa los accesos con su propio lock.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(ruta), check_same_thread=False))
