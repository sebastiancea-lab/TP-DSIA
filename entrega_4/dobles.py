"""Dobles de prueba de la Entrega 4: un LLM falso y un retriever falso, sin red."""

from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def llamada_a_tool(nombre: str, **argumentos) -> AIMessage:
    """AIMessage que pide ejecutar una tool."""
    return AIMessage(
        content="", tool_calls=[{"name": nombre, "args": argumentos, "id": f"call_{nombre}"}],
    )


class FakeLLM(BaseChatModel):
    """LLM determinista. ``guion(mensajes, n)`` devuelve la AIMessage de la llamada n.

    Sin guion responde 'Respuesta n'. Registra cuántas veces lo invocaron y qué vio.
    """

    guion: Optional[Callable[[list[BaseMessage], int], AIMessage]] = None
    llamadas: int = 0
    vistos: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-portalia"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.llamadas += 1
        self.vistos.append(list(messages))
        mensaje = (
            self.guion(list(messages), self.llamadas) if self.guion
            else AIMessage(content=f"Respuesta {self.llamadas}")
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


class FakeRetriever:
    """Retriever falso: devuelve siempre el mismo fragmento de política."""

    def __init__(self) -> None:
        self.consultas: list[str] = []

    def invoke(self, consulta: str, *args, **kwargs):
        self.consultas.append(consulta)
        return [Document(
            id="POL-DEV-001",
            page_content="El cliente puede iniciar una devolución dentro de los 10 días corridos "
                         "posteriores a la recepción del producto.",
            metadata={"categoria": "devoluciones"},
        )]
