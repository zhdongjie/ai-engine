import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.knowledge.initializer import run_init as init_knowledge_db


# ========================
# 阶段 1：数据库初始化
# ========================
def init_database():
    db_manager.init_db()

    with db_manager.session_context() as session:
        session.execute(text("SELECT 1"))

        if settings.VECTOR_STORE_TYPE.lower() == "postgresql":
            logger.info("正在激活 PGVector 插件...")
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
            logger.success("PostgreSQL & PGVector 就绪")
        else:
            logger.success("数据库连接成功")


# ========================
# 阶段 2：知识库初始化
# ========================
def init_knowledge_base():
    v_type = settings.VECTOR_STORE_TYPE.lower()

    should_init = (
            settings.INIT_KNOWLEDGE_BASE
            or (v_type != "postgresql" and not os.path.exists(settings.chroma_persist_dir))
    )

    if not should_init:
        logger.info("跳过知识库初始化")
        return

    logger.warning("开始知识库初始化...")

    try:
        init_knowledge_db()
        logger.success("知识库初始化完成")
    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")


# ========================
# 生命周期入口
# ========================
@asynccontextmanager
async def app_lifespan(_: FastAPI):
    logger.info(f"{settings.PROJECT_NAME} 启动中...")

    try:
        init_database()
        init_knowledge_base()
    except Exception as e:
        logger.critical(f"启动失败: {e}")
        raise

    yield

    logger.info(f"{settings.PROJECT_NAME} 关闭中...")
    db_manager.close_db()
