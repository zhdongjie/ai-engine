# src/ai_engine/graphs/nodes/generate.py
from langgraph.config import get_config

from ai_engine.graphs.observability import (
    get_session_id,
    observe_node,
)
from ai_engine.graphs.generator.answer import resolve_prompt_data
from ai_engine.graphs.nodes.llm_stream import stream_llm_answer
from ai_engine.graphs.state import ChatGraphState


async def generate_node(state: ChatGraphState) -> dict:
    """Generate the final RAG answer using the LLM and stream chunks via LangGraph."""
    config = get_config()
    user_input = (state.get("input") or "").strip()
    history = state.get("history") or []
    biz_type = state.get("biz_type") or "normal_chat"
    user_level = (config.get("configurable") or {}).get("user_level", "default")

    prompt_data = resolve_prompt_data(biz_type, user_level)

    context = state.get("context") or ""
    extra_data = state.get("extra_data") or {}
    sources = state.get("sources") or []

    session_id = get_session_id(config)
    async with observe_node(session_id, "generate"):
        return await stream_llm_answer(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=sources,
            config=config,
            intent="RAG",
            session_id=session_id,
            node_label="generate",
        )
