# src/ai_engine/chains/formatters/response_stream.py
from typing import AsyncIterator, Any, List

from langchain_core.messages import AIMessageChunk



# Stream Formatter
async def response_stream_formatter(
        input_stream: AsyncIterator[Any]
) -> AsyncIterator[AIMessageChunk]:
    sources: List[Any] = []
    intent = "NORMAL"
    biz_type = "normal_chat"

    async for chunk in input_stream:

        # -------- 统一兼容 chunk --------
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            additional_kwargs = chunk.get("additional_kwargs", {})
            response_metadata = chunk.get("response_metadata", {})
            chunk_id = chunk.get("id")
        else:
            content = getattr(chunk, "content", "")
            additional_kwargs = getattr(chunk, "additional_kwargs", {})
            response_metadata = getattr(chunk, "response_metadata", {})
            chunk_id = getattr(chunk, "id", None)

        if additional_kwargs:
            sources.extend(additional_kwargs.get("sources", []))
            intent = additional_kwargs.get("intent", intent)
            biz_type = additional_kwargs.get("biz_type", biz_type)

        if content:
            yield AIMessageChunk(**{
                "content": content,
                "additional_kwargs": additional_kwargs,
                "response_metadata": response_metadata,
                "id": chunk_id
            })

    sources = list({str(s): s for s in sources}.values())

    # -------- 最终收尾 chunk --------
    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "done": True,
            "sources": sources,
            "intent": intent,
            "biz_type": biz_type
        }
    })
