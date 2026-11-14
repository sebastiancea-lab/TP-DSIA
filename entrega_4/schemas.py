"""Entrega 4: contratos Pydantic de la API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# thread_id acotado: viaja a logs y a la base, no debe poder llevar basura.
PATRON_THREAD_ID = r"^[A-Za-z0-9_-]{1,64}$"


class ConsultaRequest(BaseModel):
    model_config = {"extra": "forbid"}

    pregunta: str = Field(min_length=1, description="Mensaje del cliente.")
    thread_id: Optional[str] = Field(
        default=None, pattern=PATRON_THREAD_ID,
        description="Identifica la conversación. Si se omite, se genera uno nuevo.",
    )


class ConsultaResponse(BaseModel):
    thread_id: str
    respuesta: str
    herramientas_usadas: list[str]
    bloqueado: bool = Field(
        description="True solo si el input se rechazó y el agente no corrió. "
                    "Una respuesta con PII enmascarada se entrega igual (bloqueado=false).",
    )
    motivo_bloqueo: Optional[str] = Field(
        default=None,
        description="input_injection / input_longitud (bloqueo) u output_pii "
                    "(la respuesta salió, pero con datos personales enmascarados).",
    )
    latencia_ms: float


class MensajeHistorial(BaseModel):
    rol: Literal["usuario", "asistente"]
    contenido: str


class EstadoResponse(BaseModel):
    thread_id: str
    cantidad_mensajes: int
    mensajes: list[MensajeHistorial]


class MetricasResponse(BaseModel):
    consultas_totales: int
    errores: int
    latencia_promedio_ms: float
    latencia_p95_ms: float
    herramientas: dict[str, int]
    bloqueos: dict[str, int]
    tokens_entrada: int
    tokens_salida: int


class HealthResponse(BaseModel):
    estado: Literal["ok", "degradado"]
    modelo: str
    coleccion_chroma: bool
