# src/ai_engine/infra/vector_store/chroma_provider.py
import asyncio
import os
import shutil

from langchain_chroma import Chroma

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from .base import BaseVectorStoreProvider


def _run_async(coro, action: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(f"{action} must be awaited in async context.")


class ChromaProvider(BaseVectorStoreProvider):
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._vs = None
        self._init_internal_store()

    def _init_internal_store(self):
        self._vs = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
        )

    def clear_all(self):
        return _run_async(self.aclear_all(), "clear_all")

    async def aclear_all(self) -> None:
        persist_dir = settings.chroma_persist_dir
        if os.path.exists(persist_dir):
            logger.warning(f"Deleting Chroma directory: {persist_dir}")
            await asyncio.to_thread(shutil.rmtree, persist_dir)
        self._init_internal_store()

    def delete_by_path_md5(self, path_md5: str):
        return _run_async(self.adelete_by_path_md5(path_md5), "delete_by_path_md5")

    async def adelete_by_path_md5(self, path_md5: str) -> None:
        await asyncio.to_thread(self._vs.delete, where={"path_md5": path_md5})

    def add_documents(self, docs):
        return _run_async(self.aadd_documents(docs), "add_documents")

    async def aadd_documents(self, docs) -> None:
        await asyncio.to_thread(self._vs.add_documents, docs)

    @property
    def vector_store(self):
        return self._vs
