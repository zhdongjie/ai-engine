from typing import Dict, List, Tuple

from langchain_core.documents import Document

from ai_engine.infra.db.knowledge_corpus import knowledge_corpus
from ai_engine.utils.retrieval_utils import (
    compress_context_documents,
    extract_relevant_segments,
    format_docs_with_sources,
    resolve_retrieval_runtime_config,
    select_top_documents,
)


def build_context(docs: List[Document]) -> str:
    """Build the prompt context from retrieved documents."""
    context, _ = format_docs_with_sources(docs)
    return context


async def assemble_context(
    docs: List[Document],
    biz_type: str,
) -> Tuple[List[Document], str, List[str], Dict[str, object], bool]:
    """Apply context enrichment and compression to retrieved documents."""
    runtime = resolve_retrieval_runtime_config(biz_type)
    anchor_docs = docs

    if runtime["enable_context_compression"] and anchor_docs:
        anchor_docs = select_top_documents(anchor_docs, runtime["max_context_chunks"])

    final_docs = anchor_docs
    parent_context_used = False

    if runtime["enable_small_to_big_retrieval"] and anchor_docs:
        final_docs = await knowledge_corpus.expand_to_parent_context(
            anchor_docs,
            max_parent_chunks=runtime["small_to_big_max_parent_chunks"],
            fallback_window_size=runtime["small_to_big_fallback_window_size"],
        )
        parent_context_used = bool(final_docs)
    elif runtime["enable_context_enrichment"] and anchor_docs:
        final_docs = await knowledge_corpus.expand_with_neighbors(anchor_docs, runtime["context_window_size"])

    rse_summary: Dict[str, object] = {
        "segment_count": 0,
        "retained_doc_count": len(final_docs),
        "dropped_doc_count": 0,
        "selected_segment_scores": [],
        "applied": False,
    }
    if runtime["enable_relevant_segment_extraction"] and final_docs:
        final_docs, rse_summary = extract_relevant_segments(
            final_docs,
            similarity_threshold=runtime["rse_similarity_threshold"],
            segment_score_threshold=runtime["rse_segment_score_threshold"],
            window_size=runtime["rse_window_size"],
            max_segments=runtime["rse_max_segments"],
        )

    if runtime["enable_context_compression"] and final_docs:
        final_docs = compress_context_documents(
            final_docs,
            max_chunks=runtime["max_context_chunks"],
            max_characters=runtime["max_context_characters"],
        )

    context, sources = format_docs_with_sources(final_docs)
    return final_docs, context, sources, rse_summary, parent_context_used
