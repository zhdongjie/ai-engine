# src/ai_engine/knowledge/loaders/pdf_loader.py
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .base_langchain_loader import process_langchain_directory


def process_pdf(biz_dir: Path, biz_type: str, lang: str, processor, text_splitter, mode: str) -> List[Document]:
    """处理 PDF 文件"""
    return process_langchain_directory(
        biz_dir, biz_type, lang, processor, text_splitter, mode,
        glob_pattern="**/*.pdf",
        loader_cls=PyPDFLoader
    )
