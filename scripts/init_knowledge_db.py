# scripts/init_knowledge_db.py

import os
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sqlalchemy import text

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from scripts.processors.factory import get_processor


def _build_header_path(metadata: dict) -> str:
    headers = [
        metadata.get("Header 1", "").strip(),
        metadata.get("Header 2", "").strip(),
        metadata.get("Header 3", "").strip(),
    ]
    return " > ".join(header for header in headers if header)


def _enrich_chunk_metadata(docs: List, biz_type: str, file_name: str, source_type: str) -> None:
    source_key = f"{biz_type}:{file_name}"
    total_chunks = len(docs)

    for index, doc in enumerate(docs):
        header_path = _build_header_path(doc.metadata)
        doc.metadata.update(
            {
                "biz_type": biz_type,
                "file_name": file_name,
                "source_type": source_type,
                "source_key": source_key,
                "chunk_index": index,
                "chunk_total": total_chunks,
                "header_path": header_path,
            }
        )

        if header_path:
            doc.page_content = f"[Section] {header_path}\n{doc.page_content}"


def load_documents() -> List:
    """加载并切分所有文档"""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    all_docs = []
    kb_root = Path(settings.knowledge_dir)

    if not kb_root.exists():
        logger.error(f"知识库根目录不存在: {kb_root}")
        return []

    for biz_dir in kb_root.iterdir():
        if not biz_dir.is_dir():
            continue

        biz_type = biz_dir.name
        logger.info(f"正在解析业务模块: [{biz_type}]")
        processor = get_processor(biz_type)

        # --- Markdown ---
        for md_path in biz_dir.glob("**/*.md"):
            try:
                content = md_path.read_text(encoding="utf-8")

                extracted_meta = {}
                if processor:
                    content, extracted_meta = processor.process(content, md_path)

                md_splits = md_splitter.split_text(content)
                final_splits = text_splitter.split_documents(md_splits)
                _enrich_chunk_metadata(
                    final_splits,
                    biz_type=biz_type,
                    file_name=md_path.name,
                    source_type="markdown",
                )

                for doc in final_splits:
                    doc.metadata.update(extracted_meta)
                    all_docs.append(doc)

            except Exception as e:
                logger.error(f"读取 MD 失败 {md_path}: {e}")

        # --- TXT / PDF ---
        try:
            txt_loader = DirectoryLoader(
                str(biz_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            pdf_loader = DirectoryLoader(
                str(biz_dir),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader  # type: ignore
            )

            raw_docs = txt_loader.load() + pdf_loader.load()

            for doc in raw_docs:
                source_path = Path(str(doc.metadata.get("source", "")))
                content, extracted_meta = processor.process(doc.page_content, source_path)
                doc.page_content = content
                doc.metadata.update(extracted_meta)

            splits = text_splitter.split_documents(raw_docs)
            _enrich_chunk_metadata(
                splits,
                biz_type=biz_type,
                file_name=os.path.basename(str(biz_dir)),
                source_type="file",
            )

            for doc in splits:
                if doc.page_content.strip():
                    doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "unknown"))
                    doc.metadata["source_key"] = f"{biz_type}:{doc.metadata['file_name']}"
                    all_docs.append(doc)

        except Exception as e:
            logger.error(f"TXT/PDF 解析失败: {e}")

    return all_docs


def init_pgvector(embeddings, docs):
    """初始化 PGVector"""
    logger.info("清理 PostgreSQL 旧表...")

    engine = db_manager.engine

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection"))

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="ai_knowledge_base",
        connection=engine,
        use_jsonb=True,
        create_extension=False,
    )

    logger.info(f"写入 {len(docs)} 条数据到 PGVector...")
    vector_store.add_documents(docs)

    engine.dispose()


def init_chroma(embeddings, docs):
    """初始化 Chroma"""
    persist_dir = settings.chroma_persist_dir

    if os.path.exists(persist_dir):
        logger.info(f"清理 Chroma 目录: {persist_dir}")
        shutil.rmtree(persist_dir)

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    logger.info(f"写入 {len(docs)} 条数据到 Chroma...")
    vector_store.add_documents(docs)



def run_init():
    """主入口"""
    logger.info("开始知识库初始化...")

    embeddings = OpenAIEmbeddings(
        api_key=settings.QWEN_API_KEY.get_secret_value(),
        base_url=settings.QWEN_API_BASE,
        model=settings.QWEN_MODEL_EMBEDDING,
        check_embedding_ctx_length=False,
        chunk_size=10
    )

    docs = load_documents()

    if not docs:
        logger.warning("未找到任何文档")
        return

    v_type = settings.VECTOR_STORE_TYPE.lower()
    logger.info(f"当前向量引擎: [{v_type.upper()}]")

    try:
        if v_type == "postgresql":
            init_pgvector(embeddings, docs)
        else:
            init_chroma(embeddings, docs)

        logger.success(f"知识库初始化完成！（{v_type.upper()}）")

    except Exception as e:
        logger.error(f"初始化失败: {e}")


if __name__ == "__main__":
    run_init()
