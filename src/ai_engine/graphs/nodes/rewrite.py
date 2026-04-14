import asyncio

from langgraph.config import get_config

from ai_engine.chains.common.query_transformer import transform_queries
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, log_rewrite, observe_node
from ai_engine.graphs.state import ChatGraphState


async def rewrite_node(state: ChatGraphState) -> dict:
    """Rewrite the user query to improve retrieval recall."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "rewrite"):
        if not settings.ENABLE_QUERY_REWRITE:
            log_rewrite(session_id, None)
            return {}

        input_text = (state.get("input") or "").strip()
        if not input_text:
            log_rewrite(session_id, None)
            return {}

        history = state.get("history") or []

        queries = await asyncio.to_thread(
            transform_queries,
            user_input=input_text,
            history=history,
            config=config,
        )

        rewritten_query = queries[0] if queries else input_text
        if rewritten_query.strip() == input_text:
            logger.debug("Query rewrite produced no change")
            log_rewrite(session_id, None)
            return {"rewritten_query": None}

        logger.info(f"[rewrite] session={session_id} rewritten_query={rewritten_query}")
        log_rewrite(session_id, rewritten_query)

        return {"rewritten_query": rewritten_query}
