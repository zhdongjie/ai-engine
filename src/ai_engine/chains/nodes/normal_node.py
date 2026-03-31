# src/ai_engine/chains/nodes/normal_node.py
from typing import Dict, Any, AsyncIterator

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config


async def normal_chat_run(input_data: Dict[str, Any], config: RunnableConfig) -> AsyncIterator[BaseMessage]:
    biz_type = input_data.get("biz_type", "normal_chat")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])

    logger.debug(f"进入 Normal Chat 模式 [{biz_type}]，正在加载插件...")

    prompt_data = get_prompt_config(biz_type)

    # 1. 执行插件管线
    context, extra_data = "", {}
    for plugin in get_rag_plugins(biz_type):
        context, extra_data = plugin.process([], context, extra_data, config)

    # 2. 核心渲染
    async for chunk in stream_llm_response(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=[],
            config=config,
            intent="NORMAL"
    ):
        yield chunk
