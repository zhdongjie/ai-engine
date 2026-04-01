# src/ai_engine/infra/db/pgsql.py
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ai_engine.core.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL 同步连接池管理器
    """

    def __init__(self):
        self._engine = None
        self._session_maker = None

    @property
    def engine(self):
        if self._engine is None:
            raise RuntimeError("DB not initialized")
        return self._engine

    def init_db(self) -> None:
        if self._engine is None:
            self._engine = create_engine(
                settings.sync_postgres_url,
                echo=settings.DB_ECHO,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=30,
            )

            self._session_maker = sessionmaker(
                bind=self._engine,
                class_=Session,
                expire_on_commit=False,
                autoflush=False
            )
            logger.info(f"PostgreSQL 同步连接池初始化完成 (Pool Size: {settings.DB_POOL_SIZE})")

    def close_db(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("PostgreSQL 同步连接池已安全释放")

    def get_session(self) -> Generator[Session, None, None]:
        """供 FastAPI 路由层使用的同步依赖注入"""
        if self._session_maker is None:
            raise RuntimeError("Database is not initialized. Call init_db() first.")

        with self._session_maker() as session:
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    @contextmanager
    def session_context(self) -> Generator[Session, None, None]:
        """供普通函数调用的同步上下文管理器"""
        if self._session_maker is None:
            raise RuntimeError("Database is not initialized. Call init_db() first.")

        with self._session_maker() as session:
            try:
                yield session
            except Exception as e:
                session.rollback()
                raise e


db_manager = DatabaseManager()
