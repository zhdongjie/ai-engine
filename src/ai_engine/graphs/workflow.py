from typing import AsyncIterator, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from ai_engine.core.settings import settings
from ai_engine.graphs.checkpointer import close_checkpointer, get_checkpointer
from ai_engine.graphs.nodes.context_builder import context_builder_node
from ai_engine.graphs.nodes.error_handler import error_handler_node
from ai_engine.graphs.nodes.generate import generate_node
from ai_engine.graphs.nodes.grade import grade_node
from ai_engine.graphs.nodes.multi_rag import multi_rag_node
from ai_engine.graphs.nodes.normal_chat import normal_chat_node
from ai_engine.graphs.nodes.rerank import rerank_node
from ai_engine.graphs.nodes.retrieve import retrieve_node
from ai_engine.graphs.nodes.rewrite import rewrite_node
from ai_engine.graphs.nodes.router import router_node
from ai_engine.graphs.nodes.tool_agent import tool_agent_node
from ai_engine.graphs.state import ChatGraphInput, ChatGraphState


def _route_by_biz(state: ChatGraphState) -> str:
    if state.get("use_tool_agent"):
        return "tool_agent"
    biz_type = state.get("biz_type") or "normal_chat"
    return "normal_chat" if biz_type == "normal_chat" else "rewrite"


def _should_retry(state: ChatGraphState) -> bool:
    return bool(state.get("should_retry"))


_graph_app = None
_checkpointer = None


def build_workflow():
    workflow = StateGraph(ChatGraphState, input_schema=ChatGraphInput)

    workflow.add_node("router", router_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("multi_rag", multi_rag_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("normal_chat", normal_chat_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("tool_agent", tool_agent_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        _route_by_biz,
        {
            "normal_chat": "normal_chat",
            "rewrite": "rewrite",
            "tool_agent": "tool_agent",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "multi_rag")
    workflow.add_edge("multi_rag", "rerank")
    workflow.add_edge("rerank", "grade")

    workflow.add_conditional_edges(
        "grade",
        _should_retry,
        {
            True: "rewrite",
            False: "context_builder",
        },
    )

    workflow.add_edge("context_builder", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("normal_chat", END)
    workflow.add_edge("error_handler", END)
    workflow.add_edge("tool_agent", END)

    return workflow


async def get_graph_app():
    global _graph_app
    global _checkpointer

    if _graph_app is None:
        _checkpointer = await get_checkpointer()
        workflow = build_workflow()
        _graph_app = workflow.compile(checkpointer=_checkpointer)

    return _graph_app


async def init_graph_runtime() -> None:
    if not settings.ENABLE_LANGGRAPH:
        return
    await get_graph_app()


async def shutdown_graph_runtime() -> None:
    await close_checkpointer(_checkpointer)


async def astream_graph_events(
    input_state: ChatGraphState,
    config: RunnableConfig,
) -> AsyncIterator[Dict]:
    app = await get_graph_app()
    async for event in app.astream_events(input_state, config=config):
        yield event
