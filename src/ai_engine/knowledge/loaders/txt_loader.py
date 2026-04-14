# src/ai_engine/knowledge/loaders/txt_loader.py
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from .base_langchain_loader import process_langchain_directory


async def process_txt(biz_dir: Path, biz_type: str, lang: str, processor, text_splitter, mode: str) -> List[Document]:
    """处理 TXT 文件"""
    return await process_langchain_directory(
        biz_dir, biz_type, lang, processor, text_splitter, mode,
        glob_pattern="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
