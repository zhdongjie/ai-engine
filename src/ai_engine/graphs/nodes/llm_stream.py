# src/ai_engine/graphs/nodes/llm_stream.py
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.core.logger import logger
from ai_engine.graphs.observability import (
    log_final_prompt,
    log_response,
    log_token_usage,
)


def safe_format_prompt(
    template: str,
    variables: Dict[str, Any],
    *,
    node_label: str,
    session_id: str | None = None,
) -> str:
    if not isinstance(template, str):
        logger.warning(
            f"[{node_label}] session={session_id} prompt template is not a string; skip formatting."
        )
        return str(template or "")
    try:
        return template.format_map(variables)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(
            f"[{node_label}] session={session_id} prompt format failed: {exc}."
        )
        return template


async def stream_llm_answer(
    *,
    user_input: str,
    history: List,
    biz_type: str,
    prompt_data: Dict[str, Any],
    context: str,
    extra_data: Dict[str, Any],
    sources: List,
    config: RunnableConfig,
    intent: str,
    session_id: str,
    node_label: str,
) -> Dict[str, Any]:
    writer = get_stream_writer()
    answer_parts: List[str] = []
    response_metadata: Dict[str, Any] = {}

    async for chunk in stream_llm_response(
        user_input=user_input,
        history=history,
        biz_type=biz_type,
        prompt_data=prompt_data,
        context=context,
        extra_data=extra_data,
        sources=sources,
        config=config,
        intent=intent,
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

    final_prompt = safe_format_prompt(
        prompt_data.get("content", ""),
        {
            "context": context,
            "formatted_context": context,
            "query": user_input,
            "input": user_input,
            **extra_data,
        },
        node_label=node_label,
        session_id=session_id,
    )

    logger.info(f"[{node_label}] session={session_id} final_prompt={final_prompt}")
    logger.info(f"[{node_label}] session={session_id} response={answer}")
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
