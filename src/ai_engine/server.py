# src/ai_engine/server.py
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langserve import add_routes
from sqlalchemy import text

from ai_engine.chains.chat_chain import chat_chain
from ai_engine.core.logger import logger
# 1. 引入你的核心基座
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
# 引入知识库初始化逻辑
from scripts.init_knowledge_db import run_init as init_knowledge_db


# --- A. 定义生命周期管理 ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    企业级生命周期管理：在应用启动前做检查，在关闭后做清理
    """
    logger.info(f"🚀 {settings.PROJECT_NAME} 引擎正在启动...")

    # --- 🚀 阶段 1：初始化 PostgreSQL 连接池 ---
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

    yield  # 应用在此处运行，接受 HTTP 请求

    # --- 🛑 关闭阶段：清理资源 ---
    logger.info(f"🛑 {settings.PROJECT_NAME} 正在关闭并释放资源...")
    # 👇 记得断开 PostgreSQL 连接池
    await db_manager.close_db()


# --- B. 实例化 FastAPI ---
# 此时 setup_logging 应该已经在入口处被调用过了
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

# --- C. 挂载 LangServe 路由 ---
# 这里挂载后，API 将具备流式输出、中间步骤追踪等 LangChain 原生能力
add_routes(
    app,
    chat_chain,
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    # playground_type="chat",
    playground_type="default",
)


# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "version": settings.PROJECT_VERSION,
        "vector_db": "ChromaDB Connected",
        "relational_db": "PostgreSQL Connected"
    }
