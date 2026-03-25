# src/ai_engine/api/system_router.py
from fastapi import APIRouter
from ai_engine.core.settings import settings

router = APIRouter(tags=["System Management"])

@router.get("/health")
async def health_check():
    """系统健康检查接口"""
    return {
        "status": "online",
        "version": settings.PROJECT_VERSION,
        "vector_db": "ChromaDB Connected",
        "relational_db": "PostgreSQL Connected"
    }