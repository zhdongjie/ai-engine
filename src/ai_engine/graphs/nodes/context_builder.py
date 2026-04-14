from langgraph.config import get_config

from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.core.logger import logger
from ai_engine.graphs.observability import get_session_id, observe_node
from ai_engine.graphs.retriever.context_builder import assemble_context
from ai_engine.graphs.state import ChatGraphState


async def context_builder_node(state: ChatGraphState) -> dict:
    """Build the final context for generation and apply post-processing plugins."""
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "context_builder"):
        docs = state.get("documents") or []
        biz_type = state.get("biz_type") or "normal_chat"

        if docs:
            detected_biz = docs[0].metadata.get("biz_type")
            if detected_biz and detected_biz != biz_type:
                logger.info(f"[context_builder] switching biz_type {biz_type} -> {detected_biz}")
                biz_type = detected_biz

        final_docs, context, sources, rse_summary, parent_context_used = await assemble_context(docs, biz_type)

        query = (state.get("rewritten_query") or state.get("input") or "").strip()
        extra_data = {
            "retrieval_queries": [query] if query else [],
            "retrieval_score": state.get("retrieval_score"),
            "should_retry": state.get("should_retry"),
            "parent_context_used": parent_context_used,
            "rse_summary": rse_summary,
        }

        for plugin in get_rag_plugins(biz_type):
            context, extra_data = plugin.process(final_docs, context, extra_data, config)

        logger.info(
            f"[context_builder] session={session_id} context_len={len(context)} sources={sources}"
        )

        return {
            "documents": final_docs,
            "context": context,
            "sources": sources,
            "extra_data": extra_data,
            "biz_type": biz_type,
        }
