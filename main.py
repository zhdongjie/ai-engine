# /main.py
from ai_engine.core.logger import setup_logging
from ai_engine.core.settings import settings
from ai_engine.server import app

setup_logging()

if __name__ == "__main__":
    import sys
    import uvicorn
    import asyncio

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(
        app=app,
        host=settings.PROJECT_HOST,
        port=settings.PROJECT_PORT,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    loop.run_until_complete(server.serve())
