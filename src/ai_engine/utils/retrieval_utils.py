from typing import Dict, Iterable, List, Sequence, Tuple

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.llm.llm_factory import LLMFactory


def get_doc_key(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source_key = metadata.get("source_key")
    chunk_index = metadata.get("chunk_index")
    if source_key is not None and chunk_index is not None:
        return f"{source_key}:{chunk_index}"
    file_name = metadata.get("file_name", "unknown")
    return f"{file_name}:{hash(doc.page_content)}"


def get_source_key(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source_key = metadata.get("source_key")
    if source_key is not None:
        return str(source_key)
    return str(metadata.get("file_name", "unknown"))


def _get_doc_score(doc, score_key: str = "rerank_score") -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    score = metadata.get(score_key)
    if score is None and score_key != "fusion_score":
        score = metadata.get("fusion_score")
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
