# src/ai_engine/infra/vector_store/pg_provider.py
import asyncio
from typing import List

from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from ai_engine.core.logger import logger
from ai_engine.infra.db.pgsql import db_manager
from .base import BaseVectorStoreProvider


def _run_async(coro, action: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(f"{action} must be awaited in async context.")


class PGVectorProvider(BaseVectorStoreProvider):
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._vs = None

    def _ensure_vector_store(self) -> PGVector:
        if self._vs is None:
            self._vs = PGVector(
                embeddings=self._embeddings,
                collection_name="ai_knowledge_base",
                connection=db_manager.async_engine,
                use_jsonb=True,
                create_extension=False,
                async_mode=True,
            )
        return self._vs

    @property
    def vector_store(self):
        return self._ensure_vector_store()

    def clear_all(self):
        return _run_async(self.aclear_all(), "clear_all")

    async def aclear_all(self) -> None:
        """Physically drop vector tables."""
        logger.warning("Dropping PostgreSQL vector tables...")
        async with db_manager.async_engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
        self._vs = None

    def delete_by_path_md5(self, path_md5: str):
        return _run_async(self.adelete_by_path_md5(path_md5), "delete_by_path_md5")

    async def adelete_by_path_md5(self, path_md5: str) -> None:
        """Delete old chunks by path md5."""
        _ = self._ensure_vector_store()
        async with db_manager.async_engine.begin() as conn:
            query = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'path_md5' = :p_md5")
            try:
                await conn.execute(query, {"p_md5": path_md5})
            except ProgrammingError as exc:
                logger.warning(f"Delete skipped; embedding table missing: {exc}")

    def add_documents(self, docs: List):
        return _run_async(self.aadd_documents(docs), "add_documents")

    async def aadd_documents(self, docs: List) -> None:
        """Persist documents to the vector store."""
        vector_store = self._ensure_vector_store()
        await vector_store.aadd_documents(docs)
