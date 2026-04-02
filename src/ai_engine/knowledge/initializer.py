import os
import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.knowledge.document_loader import load_documents


def init_pgvector(embeddings, docs):
    """Initialize PGVector with a full rebuild."""
    logger.info("Cleaning existing PostgreSQL vector tables")

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

    logger.info(f"Writing {len(docs)} chunks into PGVector")
    vector_store.add_documents(docs)
    engine.dispose()


def init_chroma(embeddings, docs):
    """Initialize Chroma with a full rebuild."""
    persist_dir = settings.chroma_persist_dir

    if os.path.exists(persist_dir):
        logger.info(f"Cleaning Chroma directory: {persist_dir}")
        shutil.rmtree(persist_dir)

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    logger.info(f"Writing {len(docs)} chunks into Chroma")
    vector_store.add_documents(docs)


def run_init():
    """Entry point for rebuilding the knowledge base."""
    logger.info("Starting knowledge base initialization")
    db_manager.init_db()

    embeddings = OpenAIEmbeddings(
        api_key=settings.QWEN_API_KEY.get_secret_value(),
        base_url=settings.QWEN_API_BASE,
        model=settings.QWEN_MODEL_EMBEDDING,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )

    docs = load_documents()
    if not docs:
        logger.warning("No knowledge documents were loaded")
        db_manager.close_db()
        return

    vector_store_type = settings.VECTOR_STORE_TYPE.lower()
    logger.info(f"Using vector store backend: [{vector_store_type.upper()}]")

    try:
        if vector_store_type == "postgresql":
            init_pgvector(embeddings, docs)
        else:
            init_chroma(embeddings, docs)

        logger.success(f"Knowledge base initialization completed for [{vector_store_type.upper()}]")
    except Exception as exc:
        logger.error(f"Knowledge base initialization failed: {exc}")
        raise
    finally:
        db_manager.close_db()
