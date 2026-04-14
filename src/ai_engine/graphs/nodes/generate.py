from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config, get_stream_writer

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.core.logger import logger
from ai_engine.graphs.observability import (
    get_session_id,
    log_final_prompt,
    log_response,
    log_token_usage,
    observe_node,
)
from ai_engine.graphs.generator.answer import resolve_prompt_data
from ai_engine.graphs.state import ChatGraphState


def _safe_format_prompt(template: str, variables: dict) -> str:
    try:
        return template.format_map(variables)
    except Exception:
        return template


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
        writer = get_stream_writer()
        answer_parts = []
        response_metadata = {}

        async for chunk in stream_llm_response(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=sources,
            config=config,
            intent="RAG",
        ):
            content = getattr(chunk, "content", "") or ""
            if content:
                answer_parts.append(content)
                writer({"type": "llm_chunk", "content": content})

            chunk_metadata = getattr(chunk, "additional_kwargs", None)
            if isinstance(chunk_metadata, dict) and chunk_metadata.get("done"):
                response_metadata = chunk_metadata

        answer = "".join(answer_parts).strip()
        updated_history = [*history, HumanMessage(content=user_input), AIMessage(content=answer)]

        final_prompt = _safe_format_prompt(
            prompt_data.get("content", ""),
            {
                "context": context,
                "formatted_context": context,
                "query": user_input,
                "input": user_input,
                **extra_data,
            },
        )

        logger.info(f"[generate] session={session_id} final_prompt={final_prompt}")
        logger.info(f"[generate] session={session_id} response={answer}")
        log_final_prompt(session_id, final_prompt)
        log_response(session_id, answer)
        if isinstance(response_metadata, dict):
            log_token_usage(session_id, response_metadata.get("usage_metadata"))

        writer({"type": "final_chunk", "metadata": response_metadata})

        return {
            "final_answer": answer,
            "history": updated_history,
            "response_metadata": response_metadata,
        }
