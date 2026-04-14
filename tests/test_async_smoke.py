import asyncio
import os
import shutil
import sys
import tempfile

import pytest
from langchain_core.documents import Document

from ai_engine.core.lifespan import app_lifespan
from ai_engine.core.settings import settings
from ai_engine.graphs.checkpointer import close_checkpointer, get_checkpointer
from ai_engine.infra.vector_store.chroma_provider import ChromaProvider
from ai_engine.server import app


class DummyEmbeddings:
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def _embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dim
        seed = sum(text.encode("utf-8"))
        return [((seed + i * 31) % 997) / 997.0 for i in range(self._dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


async def _run_vector_retrieval_smoke() -> None:
    original_chroma_dir = settings.CHROMA_DATA_DIR
    temp_dir = tempfile.mkdtemp(prefix="ai-engine-smoke-chroma-")
    settings.CHROMA_DATA_DIR = temp_dir
    try:
        provider = ChromaProvider(DummyEmbeddings())
        docs = [
            Document(
                page_content="hello world",
                metadata={"lang": "zh", "file_name": "smoke.txt"},
            )
        ]
        await provider.aadd_documents(docs)

        retriever = provider.vector_store.as_retriever(search_kwargs={"k": 1})
        if hasattr(retriever, "ainvoke"):
            results = await retriever.ainvoke("hello")
        else:
            results = await asyncio.to_thread(retriever.invoke, "hello")

        assert results, "vector retrieval returned no results"
    finally:
        settings.CHROMA_DATA_DIR = original_chroma_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def test_app_lifespan_smoke():
    """Integration smoke test for app startup/shutdown."""
    if os.getenv("RUN_SMOKE_TESTS") != "1":
        pytest.skip("Set RUN_SMOKE_TESTS=1 to enable integration smoke test")

    if sys.platform == "win32":
        policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy_cls is not None:
            asyncio.set_event_loop_policy(policy_cls())

    async def _run():
        async with app_lifespan(app):
            checkpointer = await get_checkpointer()
            await close_checkpointer(checkpointer)
            await _run_vector_retrieval_smoke()
            return True

    asyncio.run(_run())
