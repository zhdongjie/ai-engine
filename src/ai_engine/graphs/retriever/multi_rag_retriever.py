import asyncio
from typing import List

from langchain_core.documents import Document

from ai_engine.graphs.retriever.retrieval import retrieve_candidates


async def retrieve_multi_rag(
    query: str,
    targets: List[str],
    user_lang: str,
) -> List[Document]:
    """Retrieve documents across multiple KB targets."""
    docs: List[Document] = []
    for target in targets:
        docs.extend(await retrieve_candidates(query=query, biz_type=target, user_lang=user_lang))
    return docs
