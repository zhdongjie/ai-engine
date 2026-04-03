# src/ai_engine/chains/nodes/rag_node.py
import asyncio
from typing import Any, AsyncIterator, Dict

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.common.llm_runner import stream_llm_response
from ai_engine.chains.common.query_transformer import transform_queries
from ai_engine.chains.rag_plugins import get_rag_plugins
from ai_engine.core.kb_manager import kb_manager
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.db.knowledge_corpus import knowledge_corpus
from ai_engine.utils.retrieval_utils import (
    assess_retrieval_quality,
    collect_candidate_documents,
    compress_context_documents,
    extract_relevant_segments,
    format_docs_with_sources,
    get_reranked_docs,
    resolve_retrieval_runtime_config,
    select_top_documents,
)


async def dynamic_rag_run(input_data: Dict[str, Any], config: RunnableConfig) -> AsyncIterator[BaseMessage]:
    """Run the full retrieval and generation pipeline for RAG requests."""
    biz_type = input_data.get("biz_type", "normal_chat")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])

    configurable = config.get("configurable") or {}

    user_level = configurable.get("user_level", "default")
    user_lang = configurable.get("lang", "zh")

    kb_config = kb_manager.get_kb_config(biz_type)
    prompt_config = kb_config.get("prompt", biz_type)

    if isinstance(prompt_config, dict):
        prompt_name = prompt_config.get(user_level, prompt_config.get("default", biz_type))
    else:
        prompt_name = prompt_config

    prompt_data = get_prompt_config(prompt_name)

    runtime_config = resolve_retrieval_runtime_config(biz_type)

    queries = [user_input]
    if runtime_config["enable_query_transform"]:
        queries = transform_queries(user_input=user_input, history=history, config=config)
    logger.info(f"RAG queries prepared: {queries}")

    candidate_docs = await collect_candidate_documents(
        queries=queries,
        search_k=runtime_config["search_k"],
        lexical_k=runtime_config["lexical_k"],
        user_lang=user_lang,
        enable_lexical_retrieval=runtime_config["enable_lexical_retrieval"],
    )
    logger.info(f"RAG candidate retrieval completed with {len(candidate_docs)} chunks")

    final_docs = await asyncio.to_thread(get_reranked_docs, user_input, candidate_docs)
    logger.info(f"RAG rerank completed with {len(final_docs)} chunks")
    retrieval_quality = {
        "doc_count": len(final_docs),
        "source_count": 0,
        "top_score": 0.0,
        "tail_score": 0.0,
        "score_gap": 0.0,
        "weak_reasons": [],
        "is_confident": bool(final_docs),
        "should_retry": False,
    }
    relaxed_retrieval_used = False

    if runtime_config["enable_retrieval_quality_check"]:
        retrieval_quality = assess_retrieval_quality(final_docs)
        if retrieval_quality["should_retry"] and candidate_docs:
            relaxed_docs = await asyncio.to_thread(
                get_reranked_docs,
                user_input,
                candidate_docs,
                0.0,
                runtime_config["max_context_chunks"],
            )
            relaxed_quality = assess_retrieval_quality(relaxed_docs)

            improved_doc_count = relaxed_quality["doc_count"] > retrieval_quality["doc_count"]
            improved_top_score = relaxed_quality["top_score"] > retrieval_quality["top_score"]
            if improved_doc_count or improved_top_score:
                final_docs = relaxed_docs
                retrieval_quality = relaxed_quality
                relaxed_retrieval_used = True
                logger.info(f"Relaxed retrieval fallback applied with {len(final_docs)} chunks")

        if retrieval_quality["weak_reasons"]:
            logger.warning(
                f"RAG retrieval quality is weak: reasons={retrieval_quality['weak_reasons']}, "
                f"doc_count={retrieval_quality['doc_count']}, "
                f"source_count={retrieval_quality['source_count']}, "
                f"top_score={retrieval_quality['top_score']:.4f}"
            )

    anchor_docs = final_docs
    if runtime_config["enable_context_compression"] and anchor_docs:
        anchor_docs = select_top_documents(anchor_docs, runtime_config["max_context_chunks"])
        logger.info(f"Context anchor selection kept {len(anchor_docs)} chunks")

    parent_context_used = False
    if runtime_config["enable_small_to_big_retrieval"] and anchor_docs:
        final_docs = knowledge_corpus.expand_to_parent_context(
            anchor_docs,
            max_parent_chunks=runtime_config["small_to_big_max_parent_chunks"],
            fallback_window_size=runtime_config["small_to_big_fallback_window_size"],
        )
        parent_context_used = bool(final_docs)
        logger.info(f"Small-to-Big retrieval expanded context to {len(final_docs)} chunks")
    elif runtime_config["enable_context_enrichment"] and anchor_docs:
        final_docs = knowledge_corpus.expand_with_neighbors(anchor_docs, runtime_config["context_window_size"])
        logger.info(f"Context enrichment expanded retrieval to {len(final_docs)} chunks")
    else:
        final_docs = anchor_docs

    rse_summary = {
        "segment_count": 0,
        "retained_doc_count": len(final_docs),
        "dropped_doc_count": 0,
        "selected_segment_scores": [],
        "applied": False,
    }
    if runtime_config["enable_relevant_segment_extraction"] and final_docs:
        final_docs, rse_summary = extract_relevant_segments(
            final_docs,
            similarity_threshold=runtime_config["rse_similarity_threshold"],
            segment_score_threshold=runtime_config["rse_segment_score_threshold"],
            window_size=runtime_config["rse_window_size"],
            max_segments=runtime_config["rse_max_segments"],
        )
        logger.info(
            f"RSE retained {rse_summary['retained_doc_count']} chunks across "
            f"{rse_summary['segment_count']} segments"
        )

    if runtime_config["enable_context_compression"] and final_docs:
        final_docs = compress_context_documents(
            final_docs,
            max_chunks=runtime_config["max_context_chunks"],
            max_characters=runtime_config["max_context_characters"],
        )
        logger.info(f"Context compression kept {len(final_docs)} chunks")

    context, sources = format_docs_with_sources(final_docs)
    extra_data = {
        "retrieval_queries": queries,
        "retrieval_quality": retrieval_quality,
        "retrieval_confident": retrieval_quality.get("is_confident", bool(final_docs)),
        "relaxed_retrieval_used": relaxed_retrieval_used,
        "parent_context_used": parent_context_used,
        "rse_summary": rse_summary,
    }

    plugins = get_rag_plugins(biz_type)
    for plugin in plugins:
        context, extra_data = plugin.process(final_docs, context, extra_data, config)

    if not context.strip():
        logger.warning("RAG context is empty after retrieval and enrichment")
        yield AIMessageChunk(**{
            "content": "Sorry, no relevant reference content was found in the current knowledge base."
        })
        yield AIMessageChunk(**{
            "content": "",
            "additional_kwargs": {
                "sources": [],
                "biz_type": biz_type,
                "has_context": False,
            }
        })
        return

    if final_docs:
        new_biz_type = final_docs[0].metadata.get("biz_type", biz_type)
        if new_biz_type != biz_type:
            biz_type = new_biz_type
            logger.info(f"Switching prompt template to [{biz_type}] based on retrieval result")
            new_kb_config = kb_manager.get_kb_config(biz_type)
            new_prompt_name = new_kb_config.get("prompt", biz_type)
            prompt_data = get_prompt_config(new_prompt_name)

    async for chunk in stream_llm_response(
            user_input=user_input,
            history=history,
            biz_type=biz_type,
            prompt_data=prompt_data,
            context=context,
            extra_data=extra_data,
            sources=sources,
            config=config,
            intent="RAG",
    ):
        yield chunk
