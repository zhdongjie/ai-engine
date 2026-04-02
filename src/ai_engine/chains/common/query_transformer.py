# src/ai_engine/chains/common/query_transformer.py
import json
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings
from ai_engine.infra.llm.llm_factory import LLMFactory


def _stringify_message(message: BaseMessage) -> str | None:
    content = message.content if isinstance(message.content, str) else str(message.content)
    content = content.strip()
    if not content:
        return None

    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = "message"

    return f"{role}: {content}"


def _format_history(history: List[Any], limit: int = 6) -> str:
    lines: List[str] = []
    for message in history[-limit:]:
        if isinstance(message, BaseMessage):
            line = _stringify_message(message)
        else:
            line = str(message).strip()
        if line:
            lines.append(line)
    return "\n".join(lines) if lines else "No prior chat history."


def transform_queries(
        user_input: str,
        history: List[Any],
        config: RunnableConfig | None = None,
) -> List[str]:
    if not settings.ENABLE_QUERY_TRANSFORM:
        return [user_input]

    prompt_data = get_prompt_config("retrieval_rewrite")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_data["content"]),
            (
                "human",
                "Chat history:\n{history}\n\nOriginal user query:\n{user_input}\n\n"
                "Return JSON with a `queries` array.",
            ),
        ]
    )
    llm = LLMFactory.get_model(prompt_data.get("config", {}))
    chain = prompt_template | llm | StrOutputParser()

    try:
        raw_output = chain.invoke(
            {
                "history": _format_history(history),
                "user_input": user_input,
            },
            config=config or {},
        )
        payload = json.loads(raw_output)
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("`queries` must be a list")

        cleaned_queries: List[str] = []
        for query in queries:
            if not isinstance(query, str):
                continue
            normalized = query.strip()
            if normalized and normalized not in cleaned_queries:
                cleaned_queries.append(normalized)

        if not cleaned_queries:
            return [user_input]

        return cleaned_queries[:settings.QUERY_TRANSFORM_MAX_QUERIES]
    except Exception as e:
        logger.warning(f"Query transformation failed, fallback to original query: {e}")
        return [user_input]
