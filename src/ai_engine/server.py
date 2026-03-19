import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langserve import add_routes

# 1. 引入你的核心基座
from ai_engine.core.settings import settings
from ai_engine.core.logger import logger
from ai_engine.chains.chat_chain import chat_chain

# 引入知识库初始化逻辑
from scripts.init_knowledge_db import run_init as init_knowledge_db


# --- A. 定义生命周期管理 ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    企业级生命周期管理：在应用启动前做检查，在关闭后做清理
    """
    logger.info(f"🚀 {settings.PROJECT_NAME} 引擎正在启动...")
    logger.info(f"📂 当前项目根目录: {settings.project_root_dir}")

    # 自动初始化检查：如果本地没数据，启动时自动向量化
    if not os.path.exists(settings.chroma_persist_dir):
        logger.warning(f"📦 路径 {settings.chroma_persist_dir} 未检测到向量数据，开始执行首次初始化...")
        try:
            init_knowledge_db()
            logger.success("✅ 业务文档向量化完成，ChromaDB 已就绪！")
        except Exception as e:
            logger.critical(f"❌ 知识库初始化失败，系统无法正常提供 RAG 服务: {e}")
            # 在生产环境中，这里可以选择是否强行关闭应用
    else:
        logger.info("📦 检测到现有的 Chroma 向量数据库，跳过初始化。")

    yield  # 应用在此处运行

    logger.info(f"🛑 {settings.PROJECT_NAME} 正在关闭并释放资源...")


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
        "database": "ChromaDB Connected"
    }