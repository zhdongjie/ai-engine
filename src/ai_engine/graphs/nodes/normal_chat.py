# src/ai_engine/graphs/nodes/normal_chat.py
from langgraph.config import get_config

from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.graphs.prompts.loader import load_prompt
from ai_engine.graphs.observability import (
    get_session_id,
    log_retrieved_docs,
    log_rewrite,
    observe_node,
)
from ai_engine.graphs.nodes.llm_stream import stream_llm_answer
from ai_engine.graphs.state import ChatGraphState


async def normal_chat_node(state: ChatGraphState) -> dict:
    """Handle non-RAG chat responses."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "normal_chat"):
        user_input = (state.get("input") or "").strip()
        history = state.get("history") or []
        biz_type = state.get("biz_type") or "normal_chat"

        prompt_data = load_prompt(biz_type)

        context = ""
        extra_data = {}
        for plugin in get_rag_plugins(biz_type):
            context, extra_data = plugin.process([], context, extra_data, config)

        log_rewrite(session_id, None)
        log_retrieved_docs(session_id, [])

        return await stream_llm_answer(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=[],
            config=config,
            intent="NORMAL",
            session_id=session_id,
            node_label="normal_chat",
        )
