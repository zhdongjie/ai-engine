# src/ai_engine/server.py
from fastapi import FastAPI
from langserve import add_routes

from ai_engine.api.v1.chat_router import router as session_router
from ai_engine.api.v1.system_router import router as system_router
from ai_engine.chains.chat_chain import chat_chain
from ai_engine.core.constants import LANG_SERVE_ALLOWED_ENDPOINTS, API_V1_STR
from ai_engine.core.lifespan import app_lifespan
from ai_engine.core.openapi_config import setup_openapi
from ai_engine.core.settings import settings


def create_app() -> FastAPI:
    """应用工厂：负责将所有组件拼装成 FastAPI 实例"""

    # 1. 实例化核心 App，并挂载生命周期管理器
    fastapi_app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        lifespan=app_lifespan
    )

    # 2. 挂载自定义 RESTFul 路由
    fastapi_app.include_router(system_router, prefix=API_V1_STR)
    fastapi_app.include_router(session_router, prefix=API_V1_STR)

    if settings.ENABLE_LANGSERVE_EXTRAS:
        allowed_endpoints = None
        enable_feedback = True
        enable_trace = True
    else:
        allowed_endpoints = LANG_SERVE_ALLOWED_ENDPOINTS
        enable_feedback = False
        enable_trace = False

    # 挂载 LangServe 核心路由
    add_routes(
        fastapi_app,
        chat_chain,
        path=f"{API_V1_STR}/chat",
        enable_feedback_endpoint=enable_feedback,
        enable_public_trace_link_endpoint=enable_trace,
        enabled_endpoints=allowed_endpoints,
        playground_type="default",
    )

    # 4. 装配全局组件
    setup_openapi(fastapi_app)

    return fastapi_app


# 暴露出 app 供 Uvicorn 启动
app = create_app()
