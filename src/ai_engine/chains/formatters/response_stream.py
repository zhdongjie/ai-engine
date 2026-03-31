# src/ai_engine/chains/formatters/response_stream.py
from typing import AsyncIterator, Any, Dict

from langchain_core.messages import AIMessageChunk


async def response_stream_formatter(
        input_stream: AsyncIterator[Any]
) -> AsyncIterator[AIMessageChunk]:
    """
    流格式化拦截器：
    1. 实时转发 content。
    2. 自动聚合所有 chunk 中的 additional_kwargs，确保元数据不丢失。
    """
    # 汇总所有分片中的额外参数（如 intent, biz_type, lang, sources 等）
    aggregated_additional_kwargs: Dict[str, Any] = {
        "intent": "NORMAL",
        "biz_type": "normal_chat",
        "sources": []
    }

    # 汇总所有分片中的响应元数据（如 model_name, token_usage 等）
    aggregated_response_metadata: Dict[str, Any] = {}

    async for chunk in input_stream:
        # -------- 1. 兼容性解析（支持 字典 或 LangChain 消息对象） --------
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            current_additional_kwargs = chunk.get("additional_kwargs", {})
            current_response_metadata = chunk.get("response_metadata", {})
            chunk_id = chunk.get("id")
        else:
            content = getattr(chunk, "content", "")
            current_additional_kwargs = getattr(chunk, "additional_kwargs", {})
            current_response_metadata = getattr(chunk, "response_metadata", {})
            chunk_id = getattr(chunk, "id", None)

        # -------- 2. 核心：元数据累加（状态机模式） --------
        if current_additional_kwargs:
            # 特殊处理 sources，保持去重累加逻辑
            if "sources" in current_additional_kwargs:
                aggregated_additional_kwargs["sources"].extend(current_additional_kwargs["sources"])
                # 执行去重处理
                aggregated_additional_kwargs["sources"] = list(set(aggregated_additional_kwargs["sources"]))

            # 更新其他所有业务元数据（如 lang, model_name, usage_metadata 等）
            other_kwargs = {
                key: value for key, value in current_additional_kwargs.items()
                if key != "sources"
            }
            aggregated_additional_kwargs.update(other_kwargs)

        if current_response_metadata:
            aggregated_response_metadata.update(current_response_metadata)

        # -------- 3. 实时转发正文内容 --------
        if content:
            yield AIMessageChunk(**{
                "content": content,
                "additional_kwargs": current_additional_kwargs,
                "response_metadata": current_response_metadata,
                "id": chunk_id
            })

    # 标记流传输已完成
    aggregated_additional_kwargs["done"] = True

    # -------- 4. 最终收尾块：发送聚合后的完整元数据 --------
    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": aggregated_additional_kwargs,
        "response_metadata": aggregated_response_metadata
    })
