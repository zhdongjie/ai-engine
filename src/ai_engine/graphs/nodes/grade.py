from langgraph.config import get_config

from ai_engine.core.logger import logger
from ai_engine.graphs.observability import get_session_id, log_retrieval_quality, observe_node
from ai_engine.graphs.retriever.grade import grade_retrieval
from ai_engine.graphs.state import ChatGraphState
from ai_engine.utils.retrieval_utils import assess_retrieval_quality


async def grade_node(state: ChatGraphState) -> dict:
    """Grade retrieval quality to decide whether to retry."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "grade"):
        docs = state.get("documents") or []
        query = (state.get("rewritten_query") or state.get("input") or "").strip()

        score, should_retry = await grade_retrieval(query=query, docs=docs, config=config)

        logger.info(
            f"[grade] session={session_id} score={score:.3f} should_retry={should_retry}"
        )

        quality = assess_retrieval_quality(docs)
        log_retrieval_quality(
            session_id,
            score,
            should_retry,
            details={
                "doc_count": quality.get("doc_count"),
                "source_count": quality.get("source_count"),
                "top_score": quality.get("top_score"),
                "score_gap": quality.get("score_gap"),
                "weak_reasons": quality.get("weak_reasons"),
            },
        )

        return {"retrieval_score": score, "should_retry": should_retry}
