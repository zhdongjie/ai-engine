# src/ai_engine/infra/db/pgsql.py
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ai_engine.core.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL 连接池管理器（异步）
    """

    def __init__(self):
        self._async_engine = None
        self._async_session_maker = None

    @property
    def async_engine(self):
        if self._async_engine is None:
            raise RuntimeError("Async DB not initialized")
        return self._async_engine

    def init_db(self) -> None:
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                settings.sqlalchemy_async_url,
                echo=settings.DB_ECHO,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=30,
            )
            self._async_session_maker = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            logger.info(f"PostgreSQL 异步连接池初始化完成 (Pool Size: {settings.DB_POOL_SIZE})")

    async def close_db(self) -> None:
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session_maker = None
            logger.info("PostgreSQL 异步连接池已安全释放")

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """供 FastAPI 路由层使用的异步依赖注入"""
        if self._async_session_maker is None:
            raise RuntimeError("Async database is not initialized. Call init_db() first.")

        async with self._async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e

    @asynccontextmanager
    async def async_session_context(self) -> AsyncGenerator[AsyncSession, None]:
        """供普通函数调用的异步上下文管理器"""
        if self._async_session_maker is None:
            raise RuntimeError("Async database is not initialized. Call init_db() first.")

        async with self._async_session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e


db_manager = DatabaseManager()
