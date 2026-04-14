from langgraph.config import get_config, get_stream_writer

from ai_engine.graphs.state import ChatGraphState
from ai_engine.graphs.observability import get_session_id, log_response, observe_node


async def error_handler_node(state: ChatGraphState) -> dict:
    _ = state
    config = get_config()
    session_id = get_session_id(config)
    async with observe_node(session_id, "error_handler"):
        writer = get_stream_writer()
        message = "Sorry, something went wrong."
        writer({"type": "llm_chunk", "content": message})
        writer({"type": "final_chunk", "metadata": {"done": True, "error": True}})
        log_response(session_id, message)
        return {
            "final_answer": message,
            "response_metadata": {"done": True, "error": True},
        }
