import asyncio
from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.chains.common.query_transformer import transform_queries
from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings
from ai_engine.infra.db.knowledge_corpus import knowledge_corpus
from ai_engine.infra.db.vdb import vdb_manager
from ai_engine.utils.retrieval_utils import (
    dedupe_documents,
    format_docs_with_sources,
    get_reranked_docs,
    reciprocal_rank_fusion,
)


async def _semantic_search(query: str, search_k: int, user_lang: str) -> List:
    search_kwargs = {
        "k": search_k,
        "filter": {"lang": user_lang},
    }
    retriever = vdb_manager.store.as_retriever(search_kwargs=search_kwargs)
    docs = await asyncio.to_thread(retriever.invoke, query)

    if docs:
        return docs

    fallback_retriever = vdb_manager.store.as_retriever(search_kwargs={"k": search_k})
    return await asyncio.to_thread(fallback_retriever.invoke, query)


async def dynamic_rag_run(input_data: Dict[str, Any], config: RunnableConfig) -> AsyncIterator[BaseMessage]:
    """Run the full retrieval and generation pipeline for RAG requests."""
    biz_type = input_data.get("biz_type", "normal_chat")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])

    configurable = config.get("configurable") or {}
    user_lang = configurable.get("lang", "zh")
    prompt_data = get_prompt_config(biz_type)
    retrieval_config = prompt_data.get("retrieval_config", {})

    search_k = retrieval_config.get("k", settings.VECTOR_SEARCH_TOP_K)
    lexical_k = retrieval_config.get("lexical_k", settings.LEXICAL_SEARCH_TOP_K)
    enable_query_transform = retrieval_config.get(
        "enable_query_transform",
        settings.ENABLE_QUERY_TRANSFORM,
    )
    enable_lexical_retrieval = retrieval_config.get(
        "enable_lexical_retrieval",
        settings.ENABLE_LEXICAL_RETRIEVAL,
    )
    enable_context_enrichment = retrieval_config.get(
        "enable_context_enrichment",
        settings.ENABLE_CONTEXT_ENRICHMENT,
    )
    context_window_size = retrieval_config.get(
        "context_window_size",
        settings.CONTEXT_WINDOW_SIZE,
    )

    queries = [user_input]
    if enable_query_transform:
        queries = transform_queries(user_input=user_input, history=history, config=config)
    logger.info(f"RAG queries prepared: {queries}")

    semantic_result_sets = []
    for query in queries:
        semantic_docs = await _semantic_search(query=query, search_k=search_k, user_lang=user_lang)
        semantic_result_sets.append(semantic_docs)

    lexical_result_sets = []
    if enable_lexical_retrieval:
        for query in queries:
            lexical_docs = await asyncio.to_thread(knowledge_corpus.keyword_search, query, lexical_k)
            lexical_result_sets.append(lexical_docs)

    candidate_sets = [*semantic_result_sets, *lexical_result_sets]
    if len(candidate_sets) > 1:
        candidate_docs = reciprocal_rank_fusion(candidate_sets)
    else:
        candidate_docs = dedupe_documents(candidate_sets[0] if candidate_sets else [])

    candidate_limit = max(search_k, lexical_k) * max(1, len(queries))
    candidate_docs = candidate_docs[:candidate_limit]
    logger.info(f"RAG candidate retrieval completed with {len(candidate_docs)} chunks")

    final_docs = await asyncio.to_thread(get_reranked_docs, user_input, candidate_docs)
    logger.info(f"RAG rerank completed with {len(final_docs)} chunks")

    if enable_context_enrichment and final_docs:
        final_docs = knowledge_corpus.expand_with_neighbors(final_docs, context_window_size)
        logger.info(f"Context enrichment expanded retrieval to {len(final_docs)} chunks")

    context, sources = format_docs_with_sources(final_docs)
    extra_data = {"retrieval_queries": queries}

    plugins = get_rag_plugins(biz_type)
    for plugin in plugins:
        context, extra_data = plugin.process(final_docs, context, extra_data, config)

    if not context.strip():
        logger.warning("RAG context is empty after retrieval and enrichment")
        yield AIMessageChunk(content="抱歉，在当前的专属知识库中，没有找到与您问题相关的参考资料。")
        yield AIMessageChunk(
            content="",
            additional_kwargs={
                "sources": [],
                "biz_type": biz_type,
                "has_context": False,
            },
        )
        return

    if final_docs:
        new_biz_type = final_docs[0].metadata.get("biz_type", biz_type)
        if new_biz_type != biz_type:
            biz_type = new_biz_type
            logger.info(f"Switching prompt template to [{biz_type}] based on retrieval result")
            prompt_data = get_prompt_config(biz_type)

    async for chunk in stream_llm_response(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=sources,
            config=config,
            intent="RAG"
    ):
        yield chunk
