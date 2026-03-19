# /main.py
import uvicorn
from ai_engine.core.logger import setup_logging
from ai_engine.core.settings import settings

# 1. 第一时间初始化日志，接管所有输出
setup_logging()

if __name__ == "__main__":
    uvicorn.run(
        "ai_engine.server:app",
        host=settings.PROJECT_HOST,
        port=settings.PROJECT_PORT,
        reload=settings.PROJECT_RELOAD
    )