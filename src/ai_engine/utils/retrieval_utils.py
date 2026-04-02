# src/ai_engine/utils/retrieval_utils.py
import asyncio
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.knowledge_corpus import knowledge_corpus
from ai_engine.infra.db.vdb import vdb_manager
from ai_engine.infra.llm.llm_factory import LLMFactory
from ai_engine.utils.doc_utils import get_doc_key


def get_source_key(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source_key = metadata.get("source_key")
    if source_key is not None:
        return str(source_key)
    return str(metadata.get("file_name", "unknown"))


def resolve_retrieval_runtime_config(retrieval_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = retrieval_config or {}
    return {
        "search_k": config.get("k", settings.VECTOR_SEARCH_TOP_K),
        "lexical_k": config.get("lexical_k", settings.LEXICAL_SEARCH_TOP_K),
        "enable_query_transform": config.get(
            "enable_query_transform",
            settings.ENABLE_QUERY_TRANSFORM,
        ),
        "enable_lexical_retrieval": config.get(
            "enable_lexical_retrieval",
            settings.ENABLE_LEXICAL_RETRIEVAL,
        ),
        "enable_context_enrichment": config.get(
            "enable_context_enrichment",
            settings.ENABLE_CONTEXT_ENRICHMENT,
        ),
        "context_window_size": config.get(
            "context_window_size",
            settings.CONTEXT_WINDOW_SIZE,
        ),
        "enable_small_to_big_retrieval": config.get(
            "enable_small_to_big_retrieval",
            settings.ENABLE_SMALL_TO_BIG_RETRIEVAL,
        ),
        "small_to_big_max_parent_chunks": config.get(
            "small_to_big_max_parent_chunks",
            settings.SMALL_TO_BIG_MAX_PARENT_CHUNKS,
        ),
        "small_to_big_fallback_window_size": config.get(
            "small_to_big_fallback_window_size",
            settings.SMALL_TO_BIG_FALLBACK_WINDOW_SIZE,
        ),
        "enable_retrieval_quality_check": config.get(
            "enable_retrieval_quality_check",
            settings.ENABLE_RETRIEVAL_QUALITY_CHECK,
        ),
        "enable_relevant_segment_extraction": config.get(
            "enable_relevant_segment_extraction",
            settings.ENABLE_RELEVANT_SEGMENT_EXTRACTION,
        ),
        "rse_similarity_threshold": config.get(
            "rse_similarity_threshold",
            settings.RSE_SIMILARITY_THRESHOLD,
        ),
        "rse_segment_score_threshold": config.get(
            "rse_segment_score_threshold",
            settings.RSE_SEGMENT_SCORE_THRESHOLD,
        ),
        "rse_window_size": config.get(
            "rse_window_size",
            settings.RSE_WINDOW_SIZE,
        ),
        "rse_max_segments": config.get(
            "rse_max_segments",
            settings.RSE_MAX_SEGMENTS,
        ),
        "enable_context_compression": config.get(
            "enable_context_compression",
            settings.ENABLE_CONTEXT_COMPRESSION,
        ),
        "max_context_chunks": config.get(
            "max_context_chunks",
            settings.MAX_CONTEXT_CHUNKS,
        ),
        "max_context_characters": config.get(
            "max_context_characters",
            settings.MAX_CONTEXT_CHARACTERS,
        ),
    }


async def semantic_search(query: str, search_k: int, user_lang: str) -> List:
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


async def collect_candidate_documents(
        queries: Sequence[str],
        search_k: int,
        lexical_k: int,
        user_lang: str,
        enable_lexical_retrieval: bool = True,
) -> List:
    semantic_result_sets = []
    for query in queries:
        semantic_result_sets.append(await semantic_search(query, search_k, user_lang))

    lexical_result_sets = []
    if enable_lexical_retrieval:
        for query in queries:
            lexical_result_sets.append(await asyncio.to_thread(knowledge_corpus.keyword_search, query, lexical_k))

    candidate_sets = [*semantic_result_sets, *lexical_result_sets]
    if len(candidate_sets) > 1:
        candidate_docs = reciprocal_rank_fusion(candidate_sets)
    else:
        candidate_docs = dedupe_documents(candidate_sets[0] if candidate_sets else [])

    candidate_limit = max(search_k, lexical_k) * max(1, len(queries))
    return candidate_docs[:candidate_limit]


def _get_doc_score(doc, score_key: str = "rerank_score") -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    score = metadata.get(score_key)
    if score is None and score_key != "fusion_score":
        score = metadata.get("fusion_score")
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_anchor_chunk_index(doc) -> int:
    metadata = getattr(doc, "metadata", {}) or {}
    anchor_chunk_index = metadata.get("retrieval_anchor_chunk_index", metadata.get("chunk_index", 0))
    try:
        return int(anchor_chunk_index)
    except (TypeError, ValueError):
        return 0


def _get_chunk_index(doc) -> int:
    metadata = getattr(doc, "metadata", {}) or {}
    chunk_index = metadata.get("chunk_index", 0)
    try:
        return int(chunk_index)
    except (TypeError, ValueError):
        return 0


def _get_anchor_distance(doc) -> int:
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata.get("neighbor_distance") is not None:
        try:
            return abs(int(metadata.get("neighbor_distance", 0)))
        except (TypeError, ValueError):
            return 0
    return abs(_get_chunk_index(doc) - _get_anchor_chunk_index(doc))


def _get_segment_key(doc) -> Tuple[str, int]:
    metadata = getattr(doc, "metadata", {}) or {}
    anchor_source_key = str(metadata.get("retrieval_anchor_source_key", get_source_key(doc)))
    anchor_chunk_index = _get_anchor_chunk_index(doc)
    return anchor_source_key, anchor_chunk_index


def dedupe_documents(docs: Iterable) -> List:
    unique_docs = []
    seen = set()

    for doc in docs:
        key = get_doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def reciprocal_rank_fusion(result_sets: Sequence[Sequence], k: int | None = None) -> List:
    fused_scores = {}
    fused_docs = {}
    rrf_k = k or settings.RRF_K

    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            key = get_doc_key(doc)
            fused_docs[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_items = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    fused_results = []
    for key, score in ranked_items:
        doc = fused_docs[key]
        doc.metadata["fusion_score"] = score
        fused_results.append(doc)
    return fused_results


def get_reranked_docs(
        query: str,
        initial_docs: list,
        min_score: float | None = None,
        top_n: int | None = None,
) -> list:
    """Rerank retrieved documents with the configured rerank model."""
    if not initial_docs:
        return []

    documents_text = [doc.page_content for doc in initial_docs]
    try:
        resp = LLMFactory.call_rerank(
            query=query,
            documents=documents_text,
        )
        if resp.status_code != 200:
            logger.error(f"Rerank API error: {resp.message}")
            return initial_docs[:2]

        ranked_docs = []
        for item in resp.output.results:
            score = getattr(item, "relevance_score", item.get("relevance_score") if isinstance(item, dict) else None)
            index = getattr(item, "index", item.get("index") if isinstance(item, dict) else None)

            if index is None or score is None:
                continue

            original_doc = initial_docs[index]
            original_doc.metadata["rerank_score"] = score
            ranked_docs.append(original_doc)

        score_threshold = settings.RERANK_THRESHOLD if min_score is None else min_score
        if score_threshold is not None:
            ranked_docs = [
                doc for doc in ranked_docs
                if _get_doc_score(doc, "rerank_score") > score_threshold
            ]

        if top_n is not None:
            ranked_docs = ranked_docs[:top_n]

        return ranked_docs
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return initial_docs[:2]


def assess_retrieval_quality(docs: Sequence) -> Dict[str, object]:
    """Evaluate retrieval confidence before final generation."""
    unique_docs = dedupe_documents(docs)
    doc_count = len(unique_docs)
    source_count = len({get_source_key(doc) for doc in unique_docs})
    top_score = _get_doc_score(unique_docs[0]) if unique_docs else 0.0
    tail_score = _get_doc_score(unique_docs[-1]) if unique_docs else 0.0
    score_gap = top_score - tail_score if unique_docs else 0.0

    weak_reasons = []
    if doc_count == 0:
        weak_reasons.append("no_docs")
    if doc_count < settings.MIN_RETRIEVAL_DOCS:
        weak_reasons.append("low_doc_count")
    if source_count < settings.MIN_RETRIEVAL_SOURCES:
        weak_reasons.append("low_source_count")
    if unique_docs and top_score < settings.MIN_RERANK_SCORE:
        weak_reasons.append("low_top_score")
    if len(unique_docs) > 1 and score_gap < settings.MIN_RERANK_SCORE_GAP:
        weak_reasons.append("flat_score_distribution")

    return {
        "doc_count": doc_count,
        "source_count": source_count,
        "top_score": top_score,
        "tail_score": tail_score,
        "score_gap": score_gap,
        "weak_reasons": weak_reasons,
        "is_confident": not weak_reasons,
        "should_retry": "no_docs" in weak_reasons or "low_top_score" in weak_reasons,
    }


def select_top_documents(docs: Sequence, limit: int | None = None) -> List:
    """Keep the highest-signal chunks before context expansion."""
    unique_docs = dedupe_documents(docs)
    if not unique_docs:
        return []
    if limit is None or limit <= 0:
        return unique_docs

    ranked_docs = sorted(
        enumerate(unique_docs),
        key=lambda item: (
            _get_doc_score(item[1], "rerank_score"),
            _get_doc_score(item[1], "fusion_score"),
            -item[0],
        ),
        reverse=True,
    )
    return [doc for _, doc in ranked_docs[:limit]]


def compress_context_documents(
        docs: Sequence,
        max_chunks: int | None = None,
        max_characters: int | None = None,
) -> List:
    """Trim context size while preserving the current document order."""
    unique_docs = dedupe_documents(docs)
    if not unique_docs:
        return []

    chunk_limit = max_chunks if max_chunks and max_chunks > 0 else len(unique_docs)
    char_limit = max_characters if max_characters and max_characters > 0 else None

    compressed_docs = []
    current_chars = 0

    for doc in unique_docs:
        if len(compressed_docs) >= chunk_limit:
            break

        metadata = getattr(doc, "metadata", {}) or {}
        header_path = metadata.get("header_path", "")
        block_size = len(doc.page_content) + len(str(metadata.get("file_name", ""))) + len(str(header_path)) + 16

        if char_limit is not None and compressed_docs and current_chars + block_size > char_limit:
            continue

        compressed_docs.append(doc)
        current_chars += block_size

    return compressed_docs


def extract_relevant_segments(
        docs: Sequence,
        similarity_threshold: float | None = None,
        segment_score_threshold: float | None = None,
        window_size: int | None = None,
        max_segments: int | None = None,
) -> Tuple[List, Dict[str, object]]:
    """Filter expanded retrieval context into a smaller set of high-value segments."""
    unique_docs = dedupe_documents(docs)
    if not unique_docs:
        return [], {
            "segment_count": 0,
            "retained_doc_count": 0,
            "dropped_doc_count": 0,
            "selected_segment_scores": [],
            "applied": False,
        }

    similarity_floor = settings.RSE_SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
    segment_floor = settings.RSE_SEGMENT_SCORE_THRESHOLD if segment_score_threshold is None else segment_score_threshold
    distance_limit = settings.RSE_WINDOW_SIZE if window_size is None else window_size
    segment_limit = settings.RSE_MAX_SEGMENTS if max_segments is None else max_segments

    grouped_docs: DefaultDict[Tuple[str, int], List] = defaultdict(list)
    for doc in unique_docs:
        grouped_docs[_get_segment_key(doc)].append(doc)

    segment_items = []
    for segment_key, segment_docs in grouped_docs.items():
        filtered_docs = []
        doc_scores = []

        for doc in sorted(segment_docs, key=_get_chunk_index):
            score = max(_get_doc_score(doc, "rerank_score"), _get_doc_score(doc, "fusion_score"))
            distance = _get_anchor_distance(doc)
            if score < similarity_floor and not doc.metadata.get("is_retrieval_anchor", False):
                continue
            if distance_limit is not None and 0 <= distance_limit < distance:
                continue

            distance_penalty = max(0.0, 1.0 - (distance * 0.15))
            weighted_score = score * distance_penalty
            doc.metadata["rse_doc_score"] = round(weighted_score, 6)
            filtered_docs.append(doc)
            doc_scores.append(weighted_score)

        if not filtered_docs:
            continue

        top_score = max(doc_scores)
        avg_score = sum(doc_scores) / len(doc_scores)
        anchor_bonus = 0.05 if any(doc.metadata.get("is_retrieval_anchor", False) for doc in filtered_docs) else 0.0
        segment_score = top_score * 0.7 + avg_score * 0.3 + anchor_bonus
        if segment_score < segment_floor:
            continue

        for doc in filtered_docs:
            doc.metadata["rse_segment_score"] = round(segment_score, 6)

        segment_items.append((segment_key, segment_score, filtered_docs))

    segment_items.sort(key=lambda item: item[1], reverse=True)
    if segment_limit is not None and segment_limit > 0:
        segment_items = segment_items[:segment_limit]

    retained_docs = []
    seen = set()
    selected_scores = []
    for _, segment_score, segment_docs in segment_items:
        selected_scores.append(round(segment_score, 6))
        for doc in segment_docs:
            key = get_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            doc.metadata["rse_selected"] = True
            retained_docs.append(doc)

    retained_docs.sort(key=lambda item: (get_source_key(item), _get_chunk_index(item)))
    return retained_docs, {
        "segment_count": len(segment_items),
        "retained_doc_count": len(retained_docs),
        "dropped_doc_count": max(0, len(unique_docs) - len(retained_docs)),
        "selected_segment_scores": selected_scores,
        "applied": bool(segment_items),
    }


def summarize_retrieved_documents(docs: Sequence, limit: int | None = None) -> List[Dict[str, object]]:
    """Build a structured diagnostics view for retrieved documents."""
    unique_docs = dedupe_documents(docs)
    if limit is not None and limit > 0:
        unique_docs = unique_docs[:limit]

    summaries: List[Dict[str, object]] = []
    for doc in unique_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        summaries.append(
            {
                "source_key": str(metadata.get("source_key", "")),
                "file_name": str(metadata.get("file_name", "unknown")),
                "header_path": str(metadata.get("header_path", "")),
                "chunk_index": _get_chunk_index(doc),
                "anchor_chunk_index": _get_anchor_chunk_index(doc),
                "anchor_distance": _get_anchor_distance(doc),
                "rerank_score": round(_get_doc_score(doc, "rerank_score"), 6),
                "fusion_score": round(_get_doc_score(doc, "fusion_score"), 6),
                "rse_doc_score": round(_get_doc_score(doc, "rse_doc_score"), 6),
                "rse_segment_score": round(_get_doc_score(doc, "rse_segment_score"), 6),
                "is_retrieval_anchor": bool(metadata.get("is_retrieval_anchor", False)),
                "content_preview": doc.page_content[:160].replace("\n", " ").strip(),
            }
        )

    return summaries


def format_docs_with_sources(docs: list) -> Tuple[str, List[str]]:
    """Build the final prompt context and a de-duplicated source list."""
    if not docs:
        return "", []

    context_blocks = []
    sources = []

    for doc in dedupe_documents(docs):
        metadata = doc.metadata or {}
        file_name = metadata.get("file_name", "unknown")
        header_path = metadata.get("header_path", "").strip()
        source_label = file_name if not header_path else f"{file_name} :: {header_path}"
        context_blocks.append(f"[Source] {source_label}\n{doc.page_content}")
        sources.append(file_name)

    return "\n\n".join(context_blocks), sorted(set(sources))
