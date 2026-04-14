from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from langgraph.checkpoint.memory import InMemorySaver

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings


def _build_asyncpg_dsn() -> str:
    return settings.psycopg_dsn


def _import_async_postgres_saver():
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        return AsyncPostgresSaver
    except Exception as exc:
        raise ImportError(
            "AsyncPostgresSaver not found. Install `langgraph-checkpoint-postgres`."
        ) from exc


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def create_async_postgres_saver():
    AsyncPostgresSaver = _import_async_postgres_saver()
    dsn = _build_asyncpg_dsn()

    # Prefer a managed psycopg AsyncConnection to avoid context manager closing.
    try:
        conn = await AsyncConnection.connect(
            dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        saver = AsyncPostgresSaver(conn)
        setattr(saver, "_managed_conn", conn)
    except Exception:
        # Fallback for older saver signatures
        try:
            saver = AsyncPostgresSaver(dsn)
        except TypeError:
            pool = await asyncpg.create_pool(dsn)
            saver = AsyncPostgresSaver(pool)

    if hasattr(saver, "setup"):
        await _maybe_await(saver.setup())

    return saver


async def get_checkpointer():
    if settings.LANGGRAPH_CHECKPOINTER == "memory":
        logger.warning("LangGraph checkpointer is set to memory; not suitable for production.")
        return InMemorySaver()

    if settings.LANGGRAPH_CHECKPOINTER == "postgres":
        return await create_async_postgres_saver()

    raise ValueError(f"Unknown LANGGRAPH_CHECKPOINTER: {settings.LANGGRAPH_CHECKPOINTER}")


async def close_checkpointer(checkpointer: Any) -> None:
    if checkpointer is None:
        return
    managed_conn = getattr(checkpointer, "_managed_conn", None)
    if managed_conn is not None:
        await managed_conn.close()
        return
    if hasattr(checkpointer, "__aexit__"):
        await checkpointer.__aexit__(None, None, None)
        return
    if hasattr(checkpointer, "aclose") and inspect.iscoroutinefunction(checkpointer.aclose):
        await checkpointer.aclose()


@asynccontextmanager
async def create_pg_pool(dsn: str) -> AsyncIterator[asyncpg.Pool]:
    pool = await asyncpg.create_pool(dsn)
    try:
        yield pool
    finally:
        await pool.close()
