# src/ai_engine/core/lifespan.py
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from scripts.init_knowledge_db import run_init as init_knowledge_db


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    """
    企业级生命周期管理：集中处理组件的启动与销毁
    """
    logger.info(f"🚀 {settings.PROJECT_NAME} 引擎正在启动...")

    # --- 🚀 阶段 1：初始化 PostgreSQL ---
    try:
        db_manager.init_db()
        async with db_manager.session_context() as session:
            await session.execute(text("SELECT 1"))
        logger.success("✅ PostgreSQL 数据库连接成功")
    except Exception as e:
        logger.critical(f"❌ PostgreSQL 初始化失败: {e}")
        raise e

    # --- 🚀 阶段 2：初始化 Chroma 向量知识库 ---
    if not os.path.exists(settings.chroma_persist_dir):
        logger.warning(f"📦 未检测到向量数据，开始执行首次初始化...")
        try:
            init_knowledge_db()
            logger.success("✅ 业务文档向量化完成，ChromaDB 已就绪！")
        except Exception as e:
            logger.critical(f"❌ 知识库初始化失败: {e}")
    else:
        logger.info("📦 检测到现有的 Chroma 向量数据库，跳过初始化。")

    yield  # 🚀 服务运行中

    # --- 🛑 阶段 3：优雅关闭与资源清理 ---
    logger.info(f"🛑 {settings.PROJECT_NAME} 正在关闭并释放资源...")
    await db_manager.close_db()
