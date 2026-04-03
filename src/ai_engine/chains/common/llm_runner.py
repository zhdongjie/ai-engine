# src/ai_engine/chains/common/llm_runner.py
from typing import Dict, Any, AsyncIterator, List

from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from ai_engine.infra.llm.factory import get_llm_model


async def stream_llm_response(
        user_input: str,
        history: List[Any],
        biz_type: str,
        prompt_data: Dict[str, Any],
        context: str,
        extra_data: Dict[str, Any],
        sources: List[Any],
        config: RunnableConfig,
        intent: str = "normal"
) -> AsyncIterator[BaseMessage]:
    """统一的 LLM 执行与流式输出引擎（增强元数据版）"""

    # --- 0. 预提取配置信息 ---
    configurable = config.get("configurable") or {}
    user_lang = configurable.get("lang", "zh")

    model_config = prompt_data.get("config", {})

    model_name = model_config.get("model") or model_config.get("model_name") or "unknown"
    model_provider = model_config.get("provider", "openai")

    # 过滤掉注入指令等大块数据，只保留业务元数据存库
    clean_extra = {k: v for k, v in extra_data.items() if k != "injected_messages"}

    # 1. 初始化模型
    llm = get_llm_model(
        model_config,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    # 2. 动态组装 Messages 数组
    messages = [("system", prompt_data["content"])]

    injected_msgs = extra_data.get("injected_messages", [])
    for role, content in injected_msgs:
        messages.append((role, content))

    messages.extend([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    prompt_template = ChatPromptTemplate.from_messages(messages)

    # 3. 执行流式生成
    last_usage = None
    async for chunk in (prompt_template | llm).astream(
            {
                "input": user_input,
                "history": history,
                "context": context,
                **extra_data
            },
            config=config
    ):

        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            last_usage = chunk.usage_metadata

        yield chunk

    # --- 4. 格式化参考来源 ---
    formatted_sources = []
    if sources:
        temp_sources = []
        for item in sources:
            if isinstance(item, str):
                temp_sources.append(item)
            elif hasattr(item, "metadata"):
                name = item.metadata.get("source", item.metadata.get("title", "未知文档"))
                temp_sources.append(name)
        formatted_sources = list(set(temp_sources))

    # --- 5. 构建最终元数据块 ---
    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "biz_type": biz_type,
            "intent": intent,
            "lang": user_lang,
            "sources": formatted_sources,
            "model_name": model_name,
            "model_provider": model_provider,
            "system_prompt": prompt_data["content"],
            "has_context": bool(context.strip()),
            "usage_metadata": last_usage,
            "done": True,
            **clean_extra
        }
    })
