from typing import Iterable, List, Sequence, Tuple

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


def get_reranked_docs(query: str, initial_docs: list) -> list:
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

        final_docs = []
        for item in resp.output.results:
            score = getattr(item, "relevance_score", item.get("relevance_score") if isinstance(item, dict) else None)
            index = getattr(item, "index", item.get("index") if isinstance(item, dict) else None)

            if index is not None and score is not None and score > settings.RERANK_THRESHOLD:
                original_doc = initial_docs[index]
                original_doc.metadata["rerank_score"] = score
                final_docs.append(original_doc)
        return final_docs
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return initial_docs[:2]


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
