from langgraph.config import get_config

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, log_retrieved_docs, observe_node
from ai_engine.graphs.retriever.retrieval import retrieve_candidates, select_query
from ai_engine.graphs.state import ChatGraphState
from ai_engine.utils.retrieval_utils import summarize_retrieved_documents


async def retrieve_node(state: ChatGraphState) -> dict:
    """Retrieve candidate documents for the current query."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "retrieve"):
        input_text = state.get("input") or ""
        query = select_query(input_text, state.get("rewritten_query"))

        biz_type = state.get("biz_type") or "normal_chat"
        user_lang = (config.get("configurable") or {}).get("lang", "zh")

        docs = await retrieve_candidates(query=query, biz_type=biz_type, user_lang=user_lang)

        logger.info(f"[retrieve] session={session_id} query='{query}' docs={len(docs)}")

        if docs:
            summaries = summarize_retrieved_documents(docs, limit=settings.VECTOR_SEARCH_TOP_K)
            logger.info(f"[retrieve] session={session_id} top_docs={summaries}")

        log_retrieved_docs(session_id, docs, limit=settings.VECTOR_SEARCH_TOP_K)

        return {"documents": docs}
