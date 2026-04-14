from langgraph.config import get_config

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, log_retrieved_docs, observe_node
from ai_engine.graphs.retriever.multi_rag import resolve_multi_rag_targets
from ai_engine.graphs.retriever.multi_rag_retriever import retrieve_multi_rag
from ai_engine.graphs.retriever.retrieval import select_query
from ai_engine.graphs.state import ChatGraphState
from ai_engine.utils.retrieval_utils import dedupe_documents


async def multi_rag_node(state: ChatGraphState) -> dict:
    """Optionally expand retrieval with additional KB targets."""
    if not settings.ENABLE_MULTI_RAG:
        return {}

    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "multi_rag"):
        biz_type = state.get("biz_type") or "normal_chat"
        targets = resolve_multi_rag_targets(biz_type)
        if not targets:
            return {}

        query = select_query(state.get("input") or "", state.get("rewritten_query"))
        user_lang = (config.get("configurable") or {}).get("lang", "zh")

        extra_docs = await retrieve_multi_rag(query=query, targets=targets, user_lang=user_lang)
        if not extra_docs:
            return {}

        base_docs = state.get("documents") or []
        merged_docs = dedupe_documents([*base_docs, *extra_docs])

        logger.info(f"[multi_rag] session={session_id} targets={targets} docs={len(merged_docs)}")
        log_retrieved_docs(session_id, merged_docs)

        return {"documents": merged_docs}
