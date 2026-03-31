# src/ai_engine/chains/common/llm_runner.py
from typing import Dict, Any, AsyncIterator, List

from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from ai_engine.infra.llm.llm_factory import LLMFactory


async def stream_llm_response(
        user_input: str,
        history: List[Any],
        biz_type: str,
        prompt_data: Dict[str, Any],
        context: str,
        extra_data: Dict[str, Any],
        sources: List[Any],
        config: RunnableConfig
) -> AsyncIterator[BaseMessage]:
    """统一的 LLM 执行与流式输出引擎"""

    # 1. 初始化模型
    llm = LLMFactory.get_model(
        prompt_data.get("config", {}),
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
    async for chunk in (prompt_template | llm).astream(
            {
                "input": user_input,
                "history": history,
                "context": context,
                **extra_data
            },
            config=config
    ):
        yield chunk

    # 4. 返回标准化的结尾块
    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "sources": sources,
            "biz_type": biz_type,
            "has_context": bool(context.strip()),
            **extra_data
        }
    })
