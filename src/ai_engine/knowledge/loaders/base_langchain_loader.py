# src/ai_engine/knowledge/loaders/base_langchain_loader.py
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_engine.core.logger import logger
from ai_engine.knowledge.loader_utils import (
    group_documents_by_source,
    split_documents_for_ingestion,
    enrich_chunk_metadata,
    apply_document_augmentation
)
from ai_engine.knowledge.sync_tracker import sync_tracker


def process_langchain_directory(
        biz_dir: Path,
        biz_type: str,
        lang: str,
        processor,
        text_splitter: RecursiveCharacterTextSplitter,
        mode: str,
        glob_pattern: str,
        loader_cls,
        loader_kwargs: dict = None
) -> List[Document]:
    """通用的 LangChain DirectoryLoader 处理模板"""
    docs = []
    try:
        kwargs = loader_kwargs or {}
        loader = DirectoryLoader(
            str(biz_dir),
            glob=glob_pattern,
            loader_cls=loader_cls,
            loader_kwargs=kwargs
        )
        raw_docs = loader.load()

        filtered_raw_docs = []
        for raw_doc in raw_docs:
            source_path = Path(str(raw_doc.metadata.get("source", "")))

            action = sync_tracker.inspect_document(source_path, biz_type)
            if mode == "incremental" and action == "skip":
                continue

            content = raw_doc.page_content
            path_md5 = sync_tracker.get_path_md5(source_path)
            extracted_meta = {"lang": lang, "path_md5": path_md5}

            if processor:
                content, proc_meta = processor.process(content, source_path)
                extracted_meta.update(proc_meta)

            raw_doc.page_content = content
            raw_doc.metadata.update(extracted_meta)
            filtered_raw_docs.append(raw_doc)

        if not filtered_raw_docs:
            return docs

        grouped_docs = group_documents_by_source(filtered_raw_docs)
        for source, source_docs in grouped_docs.items():
            file_name = os.path.basename(source or "unknown")
            split_docs, chunk_strategy = split_documents_for_ingestion(source_docs, text_splitter)

            enrich_chunk_metadata(
                split_docs, biz_type=biz_type, file_name=file_name,
                source_type="file", chunk_strategy=chunk_strategy,
            )

            path_md5 = sync_tracker.get_path_md5(Path(source))
            for sd in split_docs:
                sd.metadata["path_md5"] = path_md5

            apply_document_augmentation(split_docs)

            for split_doc in split_docs:
                if split_doc.page_content.strip():
                    docs.append(split_doc)

    except Exception as exc:
        logger.error(f"Failed to parse {glob_pattern} documents for [{biz_type}]: {exc}")

    return docs
