# src/ai_engine/chains/title_chain.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.factory import get_llm_model


async def generate_session_title(user_content: str, config: RunnableConfig) -> str:
    """
    根据用户首条消息自动生成会话标题
    """
    try:
        config = config or {}
        biz_type = "session_title"

        prompt_data = get_prompt_config(biz_type)

        context = ""
        extra_data = {}
        plugins = get_rag_plugins(biz_type)

        for plugin in plugins:
            context, extra_data = plugin.process([], context, extra_data, config)

        messages = [
            ("system", prompt_data["content"])
        ]

        injected_msgs = extra_data.get("injected_messages", [])
        for role, content in injected_msgs:
            messages.append((role, content))

        messages.append(("human", "{user_content}"))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        llm = get_llm_model(prompt_data.get("config", {}))

        chain = prompt_template | llm | StrOutputParser()

        title = await chain.ainvoke({"user_content": user_content}, config=config)

        if isinstance(title, str):
            return title.replace('*', '').strip(' 。，、"\'”“\n\t')

        return "新对话"

    except Exception as e:
        logger.error(f"自动生成会话标题失败: {e}", exc_info=True)
        return "新对话"
