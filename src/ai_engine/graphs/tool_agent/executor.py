from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from ai_engine.graphs.state import ChatGraphState


class ToolAgentExecutor(Protocol):
    async def run(
        self,
        state: ChatGraphState,
        config: RunnableConfig,
    ) -> Tuple[str, Dict[str, Any]]:
        ...


def get_tool_agent_executor() -> ToolAgentExecutor | None:
    """Load the tool agent executor from infra layer if available."""
    try:
        from ai_engine.infra.tools.registry import tool_agent_executor  # type: ignore

        return tool_agent_executor
    except Exception as exc:
        logger.warning(f"[tool_agent] executor not configured: {exc}")
        return None
