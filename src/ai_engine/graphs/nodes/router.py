from langgraph.config import get_config

from ai_engine.chains.nodes.router_node import inject_kb_info, intent_router
from ai_engine.core.kb_manager import kb_manager
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.graphs.observability import get_session_id, log_query, observe_node
from ai_engine.graphs.state import ChatGraphState


async def router_node(state: ChatGraphState) -> dict:
    """Route the request to RAG or normal chat based on intent classification."""
    config = get_config()
    input_text = (state.get("input") or "").strip()
    session_id = get_session_id(config)
    async with observe_node(session_id, "router"):
        log_query(session_id, input_text)
        if not input_text:
            logger.warning("Router received empty input; defaulting to normal_chat")
            return {"intent": "NORMAL", "biz_type": "normal_chat"}

        payload = inject_kb_info({"input": input_text})
        intent_raw = await intent_router.ainvoke(payload, config=config)
        intent = str(intent_raw).strip()
        biz_type = intent if intent in kb_manager.registry else "normal_chat"

        logger.info(f"[router] session={session_id} intent={intent} biz_type={biz_type}")

        configurable = config.get("configurable") or {}
        kb_config = kb_manager.get_kb_config(biz_type)
        kb_tool_agent = False
        if isinstance(kb_config, dict):
            kb_tool_agent = bool(
                kb_config.get("enable_tool_agent")
                or kb_config.get("tool_agent", {}).get("enabled")
            )

        use_tool_agent = bool(
            settings.ENABLE_TOOL_AGENT
            and (
                configurable.get("force_tool_agent")
                or configurable.get("use_tool_agent")
                or kb_tool_agent
            )
        )

        if use_tool_agent:
            logger.info(f"[router] session={session_id} tool_agent=enabled")

        return {"intent": intent, "biz_type": biz_type, "use_tool_agent": use_tool_agent}
