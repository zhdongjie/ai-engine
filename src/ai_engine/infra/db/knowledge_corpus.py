# src/ai_engine/infra/db/knowledge_corpus.py
import asyncio
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sqlalchemy import text

from ai_engine.core.logger import logger
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.utils.doc_utils import get_doc_key

CorpusKey = Tuple[str, int]


class KnowledgeCorpusManager:
    def __init__(self):
        self._documents: Optional[List[Document]] = None
        self._bm25: Optional[BM25Retriever] = None
        self._doc_map: Dict[CorpusKey, Document] = {}
        self._source_chunks: Dict[str, List[Document]] = defaultdict(list)
        self._section_chunks: Dict[Tuple[str, str], List[Document]] = defaultdict(list)
        self._load_lock = asyncio.Lock()

    @staticmethod
    def _clone_with_retrieval_metadata(doc: Document, **extra_metadata) -> Document:
        """克隆文档并注入额外的检索元数据，确保不污染原始对象"""
        cloned_doc = deepcopy(doc)
        cloned_doc.metadata.update(extra_metadata)
        return cloned_doc

    @staticmethod
    def _iter_header_candidates(header_path: str) -> List[str]:
        """将 'A > B > C' 拆解为 ['A > B > C', 'A > B', 'A'] 用于层级匹配"""
        clean_header = header_path.strip()
        if not clean_header:
            return []
        parts = [part.strip() for part in clean_header.split(">") if part.strip()]
        return [" > ".join(parts[:index]) for index in range(len(parts), 0, -1)]

    @staticmethod
    def _select_centered_window(docs: List[Document], anchor_chunk_index: int, limit: int) -> List[Document]:
        """以目标切片为中心，选取固定数量的上下文窗口"""
        if limit <= 0 or len(docs) <= limit:
            return list(docs)

        anchor_position = 0
        for index, doc in enumerate(docs):
            if int(doc.metadata.get("chunk_index", 0)) == anchor_chunk_index:
                anchor_position = index
                break

        start = max(0, anchor_position - (limit // 2))
        end = start + limit
        if end > len(docs):
            end = len(docs)
            start = max(0, end - limit)
        return docs[start:end]

    @staticmethod
    async def _load_all_from_db() -> List[Document]:
        """直接从 PostgreSQL 向量表拉取所有已存储的切片"""
        sql = """
              SELECT e.document, e.cmetadata
              FROM langchain_pg_embedding e
                       JOIN langchain_pg_collection c ON e.collection_id = c.uuid
              WHERE c.name = 'ai_knowledge_base' \
              """
        docs = []
        try:
            async with db_manager.async_engine.connect() as conn:
                result = await conn.execute(text(sql))
                results = result.mappings().all()
                for row in results:
                    docs.append(Document(
                        page_content=row['document'],
                        metadata=row['cmetadata']
                    ))
            return docs
        except Exception as e:
            logger.error(f"从数据库提取 BM25 语料失败: {e}")
            return []

    async def _ensure_loaded(self) -> None:
        """Ensure corpus documents and indexes are loaded (lazy loading)."""
        if self._documents is not None:
            return

        async with self._load_lock:
            if self._documents is not None:
                return

            logger.info("Loading corpus documents from PostgreSQL...")

            documents = await self._load_all_from_db()

            if not documents:
                logger.warning("BM25 index build skipped: no documents found in DB.")
                self._bm25 = None
                self._documents = []
                return

            # 1. Build BM25 index
            try:
                self._bm25 = BM25Retriever.from_documents(documents)
                logger.info(f"BM25 index built ({len(documents)} documents)")
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")
                self._bm25 = None

            # 2. Build metadata maps
            self._doc_map.clear()
            self._source_chunks.clear()
            self._section_chunks.clear()
            self._documents = documents

            for doc in documents:
                metadata = doc.metadata or {}
                # Prefer source_key; fallback to path_md5
                source_key = metadata.get("source_key") or metadata.get("path_md5")
                chunk_index = metadata.get("chunk_index")

                if source_key is None or chunk_index is None:
                    continue

                key = (str(source_key), int(chunk_index))
                self._doc_map[key] = doc
                self._source_chunks[str(source_key)].append(doc)

                header_path = str(metadata.get("header_path", "")).strip()
                if header_path:
                    self._section_chunks[(str(source_key), header_path)].append(doc)

            # 3. Sort to keep ordering stable
            for docs_list in self._source_chunks.values():
                docs_list.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))
            for docs_list in self._section_chunks.values():
                docs_list.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))

            logger.info(f"Corpus loaded: {len(documents)} active chunks")

    async def keyword_search(self, query: str, top_k: int) -> List[Document]:
        """执行 BM25 关键词检索"""
        await self._ensure_loaded()
        if self._bm25 is None:
            return []
        self._bm25.k = top_k
        return list(self._bm25.invoke(query))

    async def expand_with_neighbors(self, docs: Iterable[Document], window_size: int) -> List[Document]:
        """基于 chunk_index 扩展相邻切片，并确保分数继承"""
        await self._ensure_loaded()
        if window_size <= 0:
            return list(docs)

        expanded: List[Document] = []
        seen = set()

        for doc in docs:
            metadata = doc.metadata or {}
            source_key = metadata.get("source_key") or metadata.get("path_md5")
            chunk_index = metadata.get("chunk_index")
            r_score = metadata.get("rerank_score", 0.0)
            f_score = metadata.get("fusion_score", 0.0)

            if source_key is None or chunk_index is None:
                key = get_doc_key(doc)
                if key not in seen:
                    seen.add(key)
                    expanded.append(doc)
                continue

            for offset in range(-window_size, window_size + 1):
                neighbor_key = (str(source_key), int(chunk_index) + offset)
                neighbor = self._doc_map.get(neighbor_key)
                if neighbor is None:
                    continue

                unique_key = get_doc_key(neighbor)
                if unique_key in seen:
                    continue

                seen.add(unique_key)
                expanded.append(self._clone_with_retrieval_metadata(
                    neighbor,
                    retrieval_anchor_source_key=str(source_key),
                    retrieval_anchor_chunk_index=int(chunk_index),
                    neighbor_distance=offset,
                    is_retrieval_anchor=(offset == 0),
                    rerank_score=r_score,
                    fusion_score=f_score
                ))

        expanded.sort(key=lambda x: (str(x.metadata.get("source_key", "")), int(x.metadata.get("chunk_index", 0))))
        return expanded

    async def expand_to_parent_context(
            self,
            docs: Iterable[Document],
            max_parent_chunks: int,
            fallback_window_size: int,
    ) -> List[Document]:
        """扩展到父级上下文（基于 Markdown 标题或物理窗口），并确保分数继承"""
        await self._ensure_loaded()
        expanded: List[Document] = []
        seen = set()

        for doc in docs:
            meta = doc.metadata or {}
            source_key = meta.get("source_key") or meta.get("path_md5")
            chunk_index = meta.get("chunk_index")
            header_path = str(meta.get("header_path", "")).strip()
            r_score = meta.get("rerank_score", 0.0)
            f_score = meta.get("fusion_score", 0.0)

            if source_key is None or chunk_index is None:
                expanded.append(doc)
                continue

            parent_docs: List[Document] = []
            parent_header = ""

            # 1. 尝试基于 Markdown 标题层级寻找上下文
            for cand in self._iter_header_candidates(header_path):
                cand_docs = self._section_chunks.get((str(source_key), cand), [])
                if cand_docs:
                    parent_docs = self._select_centered_window(cand_docs, int(chunk_index), max_parent_chunks)
                    parent_header = cand
                    if len(parent_docs) > 1:
                        break

            resolution = "section"
            # 2. 降级方案：如果标题层级没结果，使用 fallback_window_size 物理窗口扩展
            if not parent_docs:
                source_all_docs = self._source_chunks.get(str(source_key), [])
                if source_all_docs:
                    start_idx = max(0, int(chunk_index) - fallback_window_size)
                    end_idx = int(chunk_index) + fallback_window_size + 1
                    sub_docs = source_all_docs[start_idx:end_idx]
                    parent_docs = self._select_centered_window(sub_docs, int(chunk_index), max_parent_chunks)
                    resolution = "window"

            # 3. 合并结果并注入继承的元数据
            target_list = parent_docs if parent_docs else [doc]
            for p_doc in target_list:
                u_key = get_doc_key(p_doc)
                if u_key not in seen:
                    seen.add(u_key)
                    expanded.append(self._clone_with_retrieval_metadata(
                        p_doc,
                        retrieval_anchor_source_key=str(source_key),
                        retrieval_anchor_chunk_index=int(chunk_index),
                        retrieval_parent_resolution=resolution,
                        retrieval_parent_header_path=parent_header,
                        rerank_score=r_score,
                        fusion_score=f_score,
                        is_retrieval_anchor=int(p_doc.metadata.get("chunk_index", -1)) == int(chunk_index)
                    ))

        expanded.sort(key=lambda x: (str(x.metadata.get("source_key", "")), int(x.metadata.get("chunk_index", 0))))
        return expanded


# 导出单例对象
knowledge_corpus = KnowledgeCorpusManager()
