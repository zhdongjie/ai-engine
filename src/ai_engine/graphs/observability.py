import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.utils.retrieval_utils import summarize_retrieved_documents


def _enabled() -> bool:
    return bool(settings.ENABLE_OBS_LOGS)


@asynccontextmanager
async def observe_node(session_id: str, node_name: str) -> AsyncIterator[None]:
    if not _enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_node_latency(session_id, node_name, duration_ms)


def get_session_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable") or {}
    return str(configurable.get("session_id") or configurable.get("thread_id") or "unknown")


def log_query(session_id: str, query: str) -> None:
    if not _enabled():
        return
    logger.info(f"[obs] session_id={session_id} query={query}")


def log_rewrite(session_id: str, rewritten_query: str | None) -> None:
    if not _enabled():
        return
    logger.info(f"[obs] session_id={session_id} rewritten_query={rewritten_query}")


def log_retrieved_docs(session_id: str, docs: Sequence[Any], limit: int | None = None) -> None:
    if not _enabled():
        return
    top_k = limit if limit is not None else settings.VECTOR_SEARCH_TOP_K
    summaries = summarize_retrieved_documents(docs, limit=top_k)
    logger.info(f"[obs] session_id={session_id} retrieved_docs={summaries}")


def log_final_prompt(session_id: str, final_prompt: str) -> None:
    if not _enabled():
        return
    logger.info(f"[obs] session_id={session_id} final_prompt={final_prompt}")


def log_response(session_id: str, response: str) -> None:
    if not _enabled():
        return
    logger.info(f"[obs] session_id={session_id} response={response}")


def log_node_latency(session_id: str, node_name: str, duration_ms: float) -> None:
    if not _enabled():
        return
    logger.info(f"[metrics] session_id={session_id} node={node_name} latency_ms={duration_ms:.2f}")


def log_retrieval_quality(session_id: str, score: float, should_retry: bool, details: dict | None = None) -> None:
    if not _enabled():
        return
    payload = {
        "score": round(score, 4),
        "should_retry": bool(should_retry),
    }
    if details:
        payload.update(details)
    logger.info(f"[metrics] session_id={session_id} retrieval_quality={payload}")


def log_token_usage(session_id: str, usage: dict | None) -> None:
    if not _enabled() or not usage:
        return
    logger.info(f"[metrics] session_id={session_id} token_usage={usage}")
