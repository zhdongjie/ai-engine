# src/ai_engine/utils/doc_utils.py

from langchain_core.documents import Document

def get_doc_key(doc: Document) -> str:
    """统一获取文档的唯一键"""
    metadata = doc.metadata or {}
    source_key = metadata.get("source_key")
    chunk_index = metadata.get("chunk_index")
    if source_key is not None and chunk_index is not None:
        return f"{source_key}:{chunk_index}"
    file_name = metadata.get("file_name", "unknown")
    return f"{file_name}:{hash(doc.page_content)}"

def get_source_key(doc: Document) -> str:
    """统一获取文档的源文件键"""
    metadata = doc.metadata or {}
    source_key = metadata.get("source_key")
    if source_key is not None:
        return str(source_key)
    return str(metadata.get("file_name", "unknown"))