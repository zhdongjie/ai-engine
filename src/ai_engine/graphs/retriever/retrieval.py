import asyncio
from typing import List

from langchain_core.documents import Document

from ai_engine.utils.retrieval_utils import collect_candidate_documents, resolve_retrieval_runtime_config


async def retrieve_candidates(query: str, biz_type: str, user_lang: str) -> List[Document]:
    """Retrieve candidate documents for the given query."""
    if not query:
        return []

    runtime = resolve_retrieval_runtime_config(biz_type)
    return await collect_candidate_documents(
        queries=[query],
        search_k=runtime["search_k"],
        lexical_k=runtime["lexical_k"],
        user_lang=user_lang,
        enable_lexical_retrieval=runtime["enable_lexical_retrieval"],
    )


def select_query(state_input: str, rewritten_query: str | None) -> str:
    """Select the best available query string for retrieval."""
    return (rewritten_query or "").strip() or state_input.strip()
