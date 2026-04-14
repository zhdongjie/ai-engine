# src/ai_engine/graphs/generator/answer.py
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.core.kb_manager import kb_manager
from ai_engine.graphs.prompts.loader import load_prompt


def resolve_prompt_data(biz_type: str, user_level: str) -> Dict[str, Any]:
    """Resolve the prompt configuration for a given biz type and user level."""
    kb_config = kb_manager.get_kb_config(biz_type)
    prompt_config = kb_config.get("prompt", biz_type)

    if isinstance(prompt_config, dict):
        prompt_name = prompt_config.get(user_level, prompt_config.get("default", biz_type))
    else:
        prompt_name = prompt_config

    return load_prompt(prompt_name)


async def generate_answer(
    *,
    user_input: str,
    history: List[Any],
    biz_type: str,
    prompt_data: Dict[str, Any],
    context: str,
    extra_data: Dict[str, Any],
    sources: List[Any],
    config: RunnableConfig,
    intent: str,
) -> Tuple[str, Dict[str, Any]]:
    """Run the LLM with streaming and aggregate the final answer + metadata."""
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
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            answer_parts.append(chunk.content)

        chunk_metadata = getattr(chunk, "additional_kwargs", None)
        if isinstance(chunk_metadata, dict) and chunk_metadata.get("done"):
            response_metadata = chunk_metadata

    return "".join(answer_parts).strip(), response_metadata
