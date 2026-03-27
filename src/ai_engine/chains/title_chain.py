# src/ai_engine/chains/title_chain.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.llm_factory import LLMFactory


def generate_session_title(user_content: str) -> str:
    try:
        prompt_data = get_prompt_config("session_title")

        llm = LLMFactory.get_model(prompt_data.get("config", {}))

        prompt = PromptTemplate.from_template(prompt_data["content"])
        chain = prompt | llm | StrOutputParser()

        title = chain.invoke({"user_content": user_content})

        if isinstance(title, str):
            title = title.strip(' 。，、"\'”“\n')

        return title[:10]

    except Exception as e:
        logger.error(f"自动生成会话标题失败: {e}")
        return "新对话"
