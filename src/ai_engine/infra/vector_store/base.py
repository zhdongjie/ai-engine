# src/ai_engine/infra/vector_store/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseVectorStoreProvider(ABC):
    @abstractmethod
    def clear_all(self):
        """物理清空向量数据 (Overwrite 模式)"""
        pass

    @abstractmethod
    def delete_by_path_md5(self, path_md5: str):
        """按文件路径 MD5 删除旧切片 (Incremental 模式)"""
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document]):
        """执行数据写入"""
        pass

    @property
    @abstractmethod
    def vector_store(self):
        """暴露底层的 LangChain VectorStore 对象"""
        pass