"""Entrega 4: contratos Pydantic de la API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# El ID se guarda en SQLite y viaja a la observabilidad; se limita a caracteres
# simples y longitud acotada para mantener una clave segura y manejable.
PATRON_THREAD_ID = r"^[A-Za-z0-9_-]{1,64}$"


class ConsultaRequest(BaseModel):
    """Contrato del cuerpo que recibe POST /consultar."""

    # No se aceptan campos inesperados: un typo no debe pasar inadvertido.
    model_config = {"extra": "forbid"}

    # La pregunta es obligatoria; si no hay thread_id, la API crea una conversación nueva.
    pregunta: str = Field(min_length=1, description="Mensaje del cliente.")
    thread_id: Optional[str] = Field(
        default=None, pattern=PATRON_THREAD_ID,
        description="Identifica la conversación. Si se omite, se genera uno nuevo.",
    )


class ConsultaResponse(BaseModel):
    """Respuesta del agente junto con datos de control y observabilidad."""

    thread_id: str
    respuesta: str
    herramientas_usadas: list[str]
    bloqueado: bool = Field(
        # Distingue el rechazo del mensaje entrante de una respuesta que se pudo
        # entregar luego de enmascarar información personal.
        description="True solo si el input se rechazó y el agente no corrió. "
                    "Una respuesta con PII enmascarada se entrega igual (bloqueado=false).",
    )
    motivo_bloqueo: Optional[str] = Field(
        default=None,
        # En output_pii la respuesta sí se entrega; este campo registra que fue
        # modificada para ocultar datos personales.
        description="input_injection / input_longitud (bloqueo) u output_pii "
                    "(la respuesta salió, pero con datos personales enmascarados).",
    )
    latencia_ms: float


class MensajeHistorial(BaseModel):
    """Un mensaje visible en el historial devuelto por la API."""

    # La API expone solo estos dos roles, no mensajes internos de herramientas.
    rol: Literal["usuario", "asistente"]
    contenido: str


class EstadoResponse(BaseModel):
    """Forma de GET /estado/{thread_id}: cantidad y contenido del hilo."""

    thread_id: str
    cantidad_mensajes: int
    mensajes: list[MensajeHistorial]


class MetricasResponse(BaseModel):
    """Resumen de consultas, latencia, uso de herramientas, bloqueos y tokens."""

    consultas_totales: int
    errores: int
    latencia_promedio_ms: float
    latencia_p95_ms: float
    herramientas: dict[str, int]
    bloqueos: dict[str, int]
    tokens_entrada: int
    tokens_salida: int


class HealthResponse(BaseModel):
    """Estado de salud del servicio y disponibilidad de la colección RAG."""

    estado: Literal["ok", "degradado"]
    modelo: str
    coleccion_chroma: bool
