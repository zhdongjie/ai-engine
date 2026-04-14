import asyncio

from langgraph.config import get_config

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, observe_node
from ai_engine.graphs.state import ChatGraphState
from ai_engine.utils.retrieval_utils import get_reranked_docs


async def rerank_node(state: ChatGraphState) -> dict:
    """Optionally rerank retrieved documents."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "rerank"):
        if not settings.ENABLE_RERANK:
            return {}

        docs = state.get("documents") or []
        if not docs:
            return {}

        query = (state.get("rewritten_query") or state.get("input") or "").strip()
        if not query:
            return {}

        reranked = await asyncio.to_thread(get_reranked_docs, query, docs)

        logger.info(f"[rerank] session={session_id} docs={len(reranked)}")

        return {"documents": reranked}
