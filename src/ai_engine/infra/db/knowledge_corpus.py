from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from langchain_community.retrievers import BM25Retriever

from ai_engine.core.logger import logger
from scripts.init_knowledge_db import load_documents


CorpusKey = Tuple[str, int]


def _doc_key(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source_key = metadata.get("source_key")
    chunk_index = metadata.get("chunk_index")
    if source_key is not None and chunk_index is not None:
        return f"{source_key}:{chunk_index}"
    file_name = metadata.get("file_name", "unknown")
    return f"{file_name}:{hash(doc.page_content)}"


class KnowledgeCorpusManager:
    def __init__(self):
        self._documents = None
        self._bm25 = None
        self._doc_map: Dict[CorpusKey, object] = {}
        self._source_chunks: Dict[str, List[object]] = defaultdict(list)

    def _ensure_loaded(self) -> None:
        if self._documents is not None:
            return

        documents = load_documents()
        self._documents = documents
        self._bm25 = BM25Retriever.from_documents(documents)
        self._doc_map.clear()
        self._source_chunks.clear()

        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            source_key = metadata.get("source_key")
            chunk_index = metadata.get("chunk_index")
            if source_key is None or chunk_index is None:
                continue
            key = (str(source_key), int(chunk_index))
            self._doc_map[key] = doc
            self._source_chunks[str(source_key)].append(doc)

        for docs in self._source_chunks.values():
            docs.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))

        logger.info(f"Knowledge corpus loaded for lexical retrieval: {len(documents)} chunks")

    def keyword_search(self, query: str, top_k: int) -> List[object]:
        self._ensure_loaded()
        if self._bm25 is None:
            return []

        self._bm25.k = top_k
        return list(self._bm25.invoke(query))

    def expand_with_neighbors(self, docs: Iterable[object], window_size: int) -> List[object]:
        self._ensure_loaded()
        if window_size <= 0:
            return list(docs)

        expanded: List[object] = []
        seen = set()

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            source_key = metadata.get("source_key")
            chunk_index = metadata.get("chunk_index")

            if source_key is None or chunk_index is None:
                key = _doc_key(doc)
                if key not in seen:
                    seen.add(key)
                    expanded.append(doc)
                continue

            for offset in range(-window_size, window_size + 1):
                neighbor_key = (str(source_key), int(chunk_index) + offset)
                neighbor = self._doc_map.get(neighbor_key)
                if neighbor is None:
                    continue
                unique_key = _doc_key(neighbor)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                expanded.append(neighbor)

        expanded.sort(
            key=lambda item: (
                str(item.metadata.get("source_key", "")),
                int(item.metadata.get("chunk_index", 0)),
            )
        )
        return expanded


knowledge_corpus = KnowledgeCorpusManager()
