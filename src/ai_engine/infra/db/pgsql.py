# src/ai_engine/infra/db/pgsql.py
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)

from ai_engine.core.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL 异步连接池管理器 (企业级封装)
    """

    def __init__(self):
        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None

    def init_db(self) -> None:
        """
        初始化数据库引擎，供 FastAPI 的 lifespan 启动时调用
        """
        if self._engine is None:
            self._engine = create_async_engine(
                settings.postgres_url,
                echo=settings.DB_ECHO,

                # --- 核心连接池防御配置 ---
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,

                # 1. 连接探活：每次从池中取出连接前，先 ping 一下数据库 (防闪断)
                pool_pre_ping=True,

                # 2. 连接回收：设定连接的生命周期(秒)，防止防火墙掐断长期空闲的长连接
                pool_recycle=1800,

                # 3. 超时时间：池满时等待获取连接的最大秒数
                pool_timeout=30,
            )

            self._session_maker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False  # 生产环境建议关闭自动 flush
            )
            logger.info(f"✅ PostgreSQL 异步连接池初始化完成 (Pool Size: {settings.DB_POOL_SIZE})")

    async def close_db(self) -> None:
        """
        优雅关闭数据库连接，供 FastAPI 结束时调用，防止泄漏
        """
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            logger.info("🛑 PostgreSQL 异步连接池已安全释放")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        供 FastAPI 路由层使用的依赖注入 (Depends)
        用法: session: AsyncSession = Depends(db_manager.get_session)
        """
        if self._session_maker is None:
            raise RuntimeError("Database is not initialized. Call init_db() first.")

        async with self._session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()

    @asynccontextmanager
    async def session_context(self) -> AsyncGenerator[AsyncSession, None]:
        """
        上下文管理器：专为非 HTTP 请求（如 LangChain 适配器里的方法）设计
        用法: async with db_manager.session_context() as session:
        """
        if self._session_maker is None:
            raise RuntimeError("Database is not initialized. Call init_db() first.")

        async with self._session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()


# 暴露单例对象供全局调用
db_manager = DatabaseManager()
