# src/ai_engine/chains/nodes/normal_node.py
from typing import Dict, Any, AsyncIterator

from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.llm_factory import LLMFactory


async def normal_chat_run(input_data: Dict[str, Any]) -> AsyncIterator[BaseMessage]:
    """异步版本：处理无需查库的纯通用对话（适配 LangServe 流式输出）"""
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])
    logger.debug("进入 Normal Chat 模式，直接利用大模型本体能力回答...")

    prompt_data = get_prompt_config("normal_chat")

    llm = LLMFactory.get_model(
        prompt_data.get("config", {}),
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_data["content"]),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    async for chunk in (prompt_template | llm).astream({
        "input": user_input,
        "history": history
    }):
        yield chunk

    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "sources": [],
            "biz_type": "normal_chat",
            "has_context": False
        }
    })
