"""Entrega 4, Fase 3: agente ReAct manual con LangGraph.

agente -> (tools -> agente)* -> fin, con tope de iteraciones y nodo de cierre.
El grafo lleva el checkpointer (único dueño de la memoria); el LLM ve solo una ventana.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from entrega_4 import config  # noqa: F401  (load_dotenv antes de langchain)
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages,
)
from langgraph.errors import GraphRecursionError  # noqa: F401  (se re-exporta para la API)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from entrega_4.backend import BackendPortalia
from entrega_4.config import (
    LLM_API_KEY, LLM_MODEL, MAX_ITERACIONES, TEMPERATURA, TAGS_TRAZA, VENTANA_MENSAJES,
)
from entrega_4.prompts import MENSAJE_CORTE_ITERACIONES, SYSTEM_PROMPT
from entrega_4.tools import crear_retriever_politicas, crear_tools

def recursion_limit(max_iteraciones: int) -> int:
    """Red de seguridad por si algo se cuela: cada vuelta consume 2 pasos del grafo."""
    return max_iteraciones * 2 + 6


def mensajes_de_cierre(mensajes: list[AnyMessage], texto: str) -> list[AnyMessage]:
    """Cierra un turno: responde los tool_calls pendientes y agrega un AIMessage con ``texto``.

    Sin esto el historial quedaría inválido (tool_calls sin respuesta) para el turno siguiente.
    """
    ultimo = mensajes[-1] if mensajes else None
    pendientes = [
        ToolMessage(
            content='{"error": "Cancelada: el turno se cortó antes de ejecutarla."}',
            tool_call_id=llamada["id"], name=llamada["name"],
        )
        for llamada in (getattr(ultimo, "tool_calls", None) or [])
    ]
    return [*pendientes, AIMessage(content=texto)]


class ErrorDeConfig(RuntimeError):
    """Falta configuración obligatoria (por ejemplo LLM_API_KEY)."""


class EstadoAgente(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteraciones: int


def crear_llm():
    """LLM real con tool-calling. Se importa acá para no exigirlo en los tests."""
    if not LLM_API_KEY:
        raise ErrorDeConfig("Falta LLM_API_KEY. Copiá .env.example a .env y completala.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, api_key=LLM_API_KEY, temperature=TEMPERATURA,
    )


def ventana(mensajes: list[AnyMessage]) -> list[AnyMessage]:
    """Últimos ~VENTANA_MENSAJES mensajes, empezando en un mensaje humano.

    Garantiza que el turno en curso (último humano en adelante) nunca quede cortado.
    """
    recortados = trim_messages(
        mensajes, strategy="last", token_counter=len, max_tokens=VENTANA_MENSAJES,
        start_on="human", include_system=False,
    )
    ultimo_humano = max(
        (i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)), default=0,
    )
    turno = mensajes[ultimo_humano:]
    return recortados if len(recortados) >= len(turno) else turno


def entrada_turno(texto: str) -> dict:
    """Entrada de un turno: reinicia el contador de iteraciones (el estado persiste)."""
    return {"messages": [HumanMessage(content=texto)], "iteraciones": 0}


def config_invocacion(thread_id: str) -> dict:
    """Config de invoke: memoria por thread, metadata y tags para LangSmith."""
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "tags": list(TAGS_TRAZA),
    }


def crear_agente(
    checkpointer,
    llm=None,
    backend: Optional[BackendPortalia] = None,
    retriever=None,
    max_iteraciones: int = MAX_ITERACIONES,
):
    """Compila el grafo. ``llm``, ``backend`` y ``retriever`` se pueden inyectar (tests)."""
    llm = llm or crear_llm()
    tools = crear_tools(
        backend or BackendPortalia(), retriever or crear_retriever_politicas(),
    )
    llm_con_tools = llm.bind_tools(tools)

    def nodo_agente(estado: EstadoAgente) -> dict:
        prompt = [SystemMessage(content=SYSTEM_PROMPT), *ventana(estado["messages"])]
        respuesta = llm_con_tools.invoke(prompt)
        return {"messages": [respuesta], "iteraciones": estado.get("iteraciones", 0) + 1}

    def decidir(estado: EstadoAgente) -> str:
        ultimo = estado["messages"][-1]
        if not getattr(ultimo, "tool_calls", None):
            return END
        if estado.get("iteraciones", 0) >= max_iteraciones:
            return "cierre"
        return "tools"

    def nodo_cierre(estado: EstadoAgente) -> dict:
        """Corte controlado: cierra los tool_calls pendientes y responde con un aviso."""
        return {"messages": mensajes_de_cierre(estado["messages"], MENSAJE_CORTE_ITERACIONES)}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    grafo.add_node("cierre", nodo_cierre)
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", decidir, ["tools", "cierre", END])
    grafo.add_edge("tools", "agente")
    grafo.add_edge("cierre", END)
    # El límite sigue al max_iteraciones que se inyecta, no a la constante global.
    return grafo.compile(checkpointer=checkpointer).with_config(
        recursion_limit=recursion_limit(max_iteraciones),
    )
