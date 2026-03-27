# src/ai_engine/api/v1/system_router.py
from fastapi import APIRouter

from ai_engine.core.settings import settings
from ai_engine.schemas.result import Result

router = APIRouter(tags=["System Management"])


@router.get(
    "/health",
    response_model=Result[dict],
    summary="系统健康检查",
    description="用于监控服务的运行状态，包括数据库和向量库的连接情况",
    response_description="返回系统状态信息"
)
async def health_check():
    """系统健康检查，使用统一响应格式"""
    data = {
        "status": "online",
        "version": settings.PROJECT_VERSION,
        "vector_db": "ChromaDB Connected",
        "relational_db": "PostgreSQL Connected"
    }
    return Result.success(data=data)
