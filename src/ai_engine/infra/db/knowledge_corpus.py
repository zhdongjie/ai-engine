# src/ai_engine/infra/db/knowledge_corpus.py
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ai_engine.core.logger import logger
from ai_engine.knowledge.document_loader import load_documents
from ai_engine.utils.doc_utils import get_doc_key

CorpusKey = Tuple[str, int]


class KnowledgeCorpusManager:
    def __init__(self):
        self._documents = None
        self._bm25 = None
        self._doc_map: Dict[CorpusKey, Document] = {}
        self._source_chunks: Dict[str, List[Document]] = defaultdict(list)
        self._section_chunks: Dict[Tuple[str, str], List[Document]] = defaultdict(list)

    @staticmethod
    def _clone_with_retrieval_metadata(doc: Document, **extra_metadata) -> Document:
        cloned_doc = deepcopy(doc)
        cloned_doc.metadata.update(extra_metadata)
        return cloned_doc

    @staticmethod
    def _iter_header_candidates(header_path: str) -> List[str]:
        clean_header = header_path.strip()
        if not clean_header:
            return []

        parts = [part.strip() for part in clean_header.split(">") if part.strip()]
        return [" > ".join(parts[:index]) for index in range(len(parts), 0, -1)]

    @staticmethod
    def _select_centered_window(docs: List[Document], anchor_chunk_index: int, limit: int) -> List[Document]:
        if limit <= 0 or len(docs) <= limit:
            return list(docs)

        anchor_position = 0
        for index, doc in enumerate(docs):
            chunk_index = int(doc.metadata.get("chunk_index", 0))
            if chunk_index == anchor_chunk_index:
                anchor_position = index
                break

        start = max(0, anchor_position - (limit // 2))
        end = start + limit
        if end > len(docs):
            end = len(docs)
            start = max(0, end - limit)
        return docs[start:end]

    def _ensure_loaded(self) -> None:
        if self._documents is not None:
            return

        documents = load_documents()
        self._documents = documents
        self._bm25 = BM25Retriever.from_documents(documents)
        self._doc_map.clear()
        self._source_chunks.clear()
        self._section_chunks.clear()

        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            source_key = metadata.get("source_key")
            chunk_index = metadata.get("chunk_index")
            if source_key is None or chunk_index is None:
                continue
            key = (str(source_key), int(chunk_index))
            self._doc_map[key] = doc
            self._source_chunks[str(source_key)].append(doc)

            header_path = str(metadata.get("header_path", "")).strip()
            if header_path:
                self._section_chunks[(str(source_key), header_path)].append(doc)

        for docs in self._source_chunks.values():
            docs.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))
        for docs in self._section_chunks.values():
            docs.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))

        logger.info(f"Knowledge corpus loaded for lexical retrieval: {len(documents)} chunks")

    def keyword_search(self, query: str, top_k: int) -> List[Document]:
        self._ensure_loaded()
        if self._bm25 is None:
            return []

        self._bm25.k = top_k
        return list(self._bm25.invoke(query))

    def expand_with_neighbors(self, docs: Iterable[Document], window_size: int) -> List[Document]:
        self._ensure_loaded()
        if window_size <= 0:
            return list(docs)

        expanded: List[Document] = []
        seen = set()

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            source_key = metadata.get("source_key")
            chunk_index = metadata.get("chunk_index")

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
                enriched_neighbor = self._clone_with_retrieval_metadata(
                    neighbor,
                    retrieval_anchor_source_key=str(source_key),
                    retrieval_anchor_chunk_index=int(chunk_index),
                    neighbor_distance=offset,
                    is_retrieval_anchor=offset == 0,
                )
                expanded.append(enriched_neighbor)

        expanded.sort(
            key=lambda item: (
                str(item.metadata.get("source_key", "")),
                int(item.metadata.get("chunk_index", 0)),
            )
        )
        return expanded

    def expand_to_parent_context(
            self,
            docs: Iterable[Document],
            max_parent_chunks: int,
            fallback_window_size: int,
    ) -> List[Document]:
        self._ensure_loaded()

        expanded: List[Document] = []
        seen = set()

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            source_key = metadata.get("source_key")
            chunk_index = metadata.get("chunk_index")
            header_path = str(metadata.get("header_path", "")).strip()

            if source_key is None or chunk_index is None:
                unique_key = get_doc_key(doc)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                expanded.append(doc)
                continue

            parent_docs: List[Document] = []
            parent_header = ""
            for candidate_header in self._iter_header_candidates(header_path):
                candidate_docs = self._section_chunks.get((str(source_key), candidate_header), [])
                if not candidate_docs:
                    continue
                parent_docs = self._select_centered_window(candidate_docs, int(chunk_index), max_parent_chunks)
                parent_header = candidate_header
                if len(parent_docs) > 1:
                    break

            resolution = "section"
            if not parent_docs:
                source_docs = self._source_chunks.get(str(source_key), [])
                if source_docs:
                    start_index = max(0, int(chunk_index) - max(0, fallback_window_size))
                    end_index = int(chunk_index) + max(0, fallback_window_size) + 1
                    parent_docs = source_docs[start_index:end_index]
                    parent_docs = self._select_centered_window(parent_docs, int(chunk_index), max_parent_chunks)
                    resolution = "window"

            if not parent_docs:
                unique_key = get_doc_key(doc)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                expanded.append(doc)
                continue

            for parent_doc in parent_docs:
                unique_key = get_doc_key(parent_doc)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                expanded.append(
                    self._clone_with_retrieval_metadata(
                        parent_doc,
                        retrieval_anchor_source_key=str(source_key),
                        retrieval_anchor_chunk_index=int(chunk_index),
                        retrieval_parent_resolution=resolution,
                        retrieval_parent_header_path=parent_header,
                        is_retrieval_anchor=int(parent_doc.metadata.get("chunk_index", -1)) == int(chunk_index),
                    )
                )

        expanded.sort(
            key=lambda item: (
                str(item.metadata.get("source_key", "")),
                int(item.metadata.get("chunk_index", 0)),
            )
        )
        return expanded


knowledge_corpus = KnowledgeCorpusManager()
