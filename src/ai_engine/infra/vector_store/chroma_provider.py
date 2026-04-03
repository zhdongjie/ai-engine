# src/ai_engine/infra/vector_store/chroma_provider.py
import os
import shutil

from langchain_chroma import Chroma

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from .base import BaseVectorStoreProvider


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
        persist_dir = settings.chroma_persist_dir
        if os.path.exists(persist_dir):
            logger.warning(f"正在删除 Chroma 目录: {persist_dir}")
            shutil.rmtree(persist_dir)
        self._init_internal_store()

    def delete_by_path_md5(self, path_md5: str):
        self._vs.delete(where={"path_md5": path_md5})

    def add_documents(self, docs):
        self._vs.add_documents(docs)

    @property
    def vector_store(self):
        return self._vs
