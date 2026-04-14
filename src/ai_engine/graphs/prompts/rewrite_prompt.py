from typing import Dict

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from ai_engine.graphs.prompts.loader import load_prompt
from ai_engine.core.settings import settings
from ai_engine.infra.llm.factory import get_llm_model


async def rewrite_query(query: str, history_text: str, config: RunnableConfig) -> str:
    """Rewrite a single query using the configured rewrite prompt."""
    if not settings.ENABLE_QUERY_REWRITE:
        return query

    prompt_data = load_prompt("retrieval_rewrite")
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
    llm = get_llm_model(prompt_data.get("config", {}))
    chain = prompt_template | llm | StrOutputParser()

    try:
        raw_output = await chain.ainvoke(
            {
                "history": history_text,
                "user_input": query,
            },
            config=config,
        )
        data = DictParser.parse(raw_output)
        queries = data.get("queries", [])
        if isinstance(queries, list) and queries:
            return str(queries[0]).strip() or query
    except Exception as exc:
        logger.warning(f"Rewrite prompt failed, fallback to original query: {exc}")

    return query


class DictParser:
    @staticmethod
    def parse(raw: str) -> Dict[str, object]:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
