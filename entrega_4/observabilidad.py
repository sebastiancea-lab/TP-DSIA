"""Entrega 4, Fase 6: logs JSON y métricas en memoria (thread-safe)."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from entrega_4.config import LOG_LEVEL

NOMBRE_LOGGER = "portalia"


class FormatterJSON(logging.Formatter):
    """Una línea JSON por evento: ts, nivel, evento y los campos extra."""

    def format(self, record: logging.LogRecord) -> str:
        linea = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "nivel": record.levelname,
            "evento": record.getMessage(),
            **getattr(record, "datos", {}),
        }
        return json.dumps(linea, ensure_ascii=False, default=str)


def obtener_logger() -> logging.Logger:
    """Logger 'portalia' con salida JSON a stderr. Idempotente."""
    logger = logging.getLogger(NOMBRE_LOGGER)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(FormatterJSON())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def log_evento(evento: str, nivel: int = logging.INFO, **datos) -> None:
    obtener_logger().log(nivel, evento, extra={"datos": datos})


def mensajes_del_turno(mensajes: Iterable[AnyMessage]) -> list[AnyMessage]:
    """Mensajes generados desde el último mensaje humano (el turno en curso)."""
    mensajes = list(mensajes)
    inicio = max((i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)), default=-1)
    return mensajes[inicio + 1:]


def herramientas_del_turno(nuevos: Iterable[AnyMessage]) -> list[str]:
    """Nombres de las tools que pidió el LLM, leídos de AIMessage.tool_calls."""
    return [
        llamada["name"]
        for mensaje in nuevos if isinstance(mensaje, AIMessage)
        for llamada in (mensaje.tool_calls or [])
    ]


def tokens_del_turno(nuevos: Iterable[AnyMessage]) -> tuple[int, int]:
    """Tokens (entrada, salida) sumados de usage_metadata de cada llamada al LLM."""
    entrada = salida = 0
    for mensaje in nuevos:
        uso = getattr(mensaje, "usage_metadata", None) if isinstance(mensaje, AIMessage) else None
        if uso:
            entrada += uso.get("input_tokens", 0)
            salida += uso.get("output_tokens", 0)
    return entrada, salida


def percentil(valores: list[float], p: float) -> float:
    """Percentil por orden (nearest-rank). 0.0 si no hay datos."""
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


class Metricas:
    """Contadores del servicio. Un lock protege el estado: los endpoints corren en threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consultas = 0
        self._errores = 0
        self._latencias: list[float] = []
        self._herramientas: Counter[str] = Counter()
        self._bloqueos: Counter[str] = Counter()
        self._tokens_entrada = 0
        self._tokens_salida = 0

    def registrar_consulta(
        self, latencia_ms: float, herramientas: Iterable[str] = (), error: bool = False,
        tokens: tuple[int, int] = (0, 0),
    ) -> None:
        with self._lock:
            self._consultas += 1
            self._errores += int(error)
            self._latencias.append(latencia_ms)
            self._herramientas.update(herramientas)
            self._tokens_entrada += tokens[0]
            self._tokens_salida += tokens[1]

    def registrar_bloqueo(self, tipo: str) -> None:
        with self._lock:
            self._bloqueos[tipo] += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencias = list(self._latencias)
            promedio = sum(latencias) / len(latencias) if latencias else 0.0
            return {
                "consultas_totales": self._consultas,
                "errores": self._errores,
                "latencia_promedio_ms": round(promedio, 2),
                "latencia_p95_ms": round(percentil(latencias, 95), 2),
                "herramientas": dict(self._herramientas),
                "bloqueos": dict(self._bloqueos),
                "tokens_entrada": self._tokens_entrada,
                "tokens_salida": self._tokens_salida,
            }
