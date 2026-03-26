# src/ai_engine/chains/title_chain.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings


async def agenerate_session_title(user_content: str) -> str:
    try:
        prompt_data = get_prompt_config("session_title")
        prompt_config = prompt_data.get("config", {})

        llm = ChatOpenAI(
            model=prompt_config.get("model", settings.QWEN_MODEL_LLM),
            api_key=settings.QWEN_API_KEY.get_secret_value(),
            base_url=settings.QWEN_API_BASE,
            temperature=prompt_config.get("temperature", 0.3),
            max_tokens=prompt_config.get("max_tokens", 20)
        )

        prompt = PromptTemplate.from_template(prompt_data["content"])
        chain = prompt | llm | StrOutputParser()

        title = await chain.ainvoke({"user_content": user_content})

        clean_title = title.strip(' 。，、"\'”“\n')
        return clean_title[:10]

    except Exception as e:
        logger.error(f"自动生成会话标题失败: {e}")
        return "新对话"
