# src/ai_engine/knowledge/loaders/markdown_loader.py
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ai_engine.core.logger import logger
from ai_engine.knowledge.loader_utils import (
    split_documents_for_ingestion,
    enrich_chunk_metadata,
    apply_document_augmentation
)
from ai_engine.knowledge.sync_tracker import sync_tracker


def process_markdown(
        biz_dir: Path, biz_type: str, lang: str, processor,
        text_splitter: RecursiveCharacterTextSplitter,
        markdown_splitter: MarkdownHeaderTextSplitter, mode: str
) -> List[Document]:
    docs = []
    for markdown_path in biz_dir.glob("**/*.md"):
        try:
            action = sync_tracker.inspect_document(markdown_path, biz_type)
            if mode == "incremental" and action == "skip":
                continue

            path_md5 = sync_tracker.get_path_md5(markdown_path)
            content = markdown_path.read_text(encoding="utf-8")
            extracted_meta = {"lang": lang, "path_md5": path_md5}

            if processor:
                content, proc_meta = processor.process(content, markdown_path)
                extracted_meta.update(proc_meta)

            markdown_splits = markdown_splitter.split_text(content)
            final_splits, chunk_strategy = split_documents_for_ingestion(markdown_splits, text_splitter)

            enrich_chunk_metadata(
                final_splits, biz_type=biz_type, file_name=markdown_path.name,
                source_type="markdown", chunk_strategy=chunk_strategy,
            )

            for doc in final_splits:
                doc.metadata.update(extracted_meta)

            apply_document_augmentation(final_splits)
            docs.extend(final_splits)
        except Exception as exc:
            logger.error(f"Failed to parse markdown file {markdown_path}: {exc}")
    return docs
