from langgraph.config import get_config, get_stream_writer

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, log_response, observe_node
from ai_engine.graphs.state import ChatGraphState
from ai_engine.graphs.tool_agent.executor import get_tool_agent_executor


async def tool_agent_node(state: ChatGraphState) -> dict:
    """Placeholder for future tool-agent execution."""
    config = get_config()
    session_id = get_session_id(config)
    _ = state

    async with observe_node(session_id, "tool_agent"):
        if not settings.ENABLE_TOOL_AGENT:
            logger.warning(f"[tool_agent] session={session_id} disabled; returning empty response")
            return {}

        executor = get_tool_agent_executor()
        if executor is None:
            message = "Tool agent is not configured yet."
            metadata = {"done": True, "tool_agent": True, "error": True}
        else:
            message, metadata = await executor.run(state, config)
            if not isinstance(metadata, dict):
                metadata = {"done": True, "tool_agent": True}
            else:
                metadata = {"done": True, "tool_agent": True, **metadata}

        writer = get_stream_writer()
        writer({"type": "llm_chunk", "content": message})
        writer({"type": "final_chunk", "metadata": metadata})

        log_response(session_id, message)

        return {
            "final_answer": message,
            "response_metadata": metadata,
        }
