import json
from typing import Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from ai_engine.graphs.prompts.loader import load_prompt
from ai_engine.infra.llm.factory import get_llm_model
from ai_engine.utils.retrieval_utils import (
    assess_retrieval_quality,
    format_docs_with_sources,
    select_top_documents,
)


async def grade_retrieval(query: str, docs: list, config: RunnableConfig) -> Tuple[float, bool]:
    """Grade retrieval quality with an LLM, falling back to deterministic checks."""
    if not docs:
        return 0.0, True

    prompt_data = load_prompt("retrieval_eval")
    prompt = PromptTemplate.from_template(prompt_data["content"])
    llm = get_llm_model(prompt_data.get("config", {}))
    chain = prompt | llm | StrOutputParser()

    top_docs = select_top_documents(docs, limit=3)
    retrieved_context, _ = format_docs_with_sources(top_docs)

    raw_output = None
    try:
        raw_output = await chain.ainvoke(
            {
                "query": query,
                "retrieved_context": retrieved_context,
            },
            config=config,
        )
        payload = json.loads(raw_output)
        score_raw = payload.get("score", 0)
        try:
            score_value = float(score_raw)
        except (TypeError, ValueError):
            score_value = 0.0

        score = max(0.0, min(1.0, score_value / 10.0))
        is_relevant = str(payload.get("is_relevant", "")).strip().upper() == "YES"
        should_retry = (not is_relevant) or score < 0.5
        return score, should_retry
    except Exception as exc:
        logger.warning(f"Retrieval grading failed, fallback to heuristic check: {exc}")
        quality = assess_retrieval_quality(docs)
        score = float(min(1.0, max(0.0, quality.get("top_score", 0.0))))
        should_retry = bool(quality.get("should_retry", False))
        return score, should_retry
    finally:
        if raw_output:
            logger.debug(f"Retrieval grader raw output: {raw_output}")
