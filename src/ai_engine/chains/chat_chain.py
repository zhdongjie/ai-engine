# src/ai_engine/chains/chat_chain.py
import uuid

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda, ConfigurableFieldSpec, RunnablePassthrough, Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from ai_engine.chains.nodes.router_node import intent_router, route_logger, route_logic
from ai_engine.core.logger import logger
from ai_engine.infra.llm.message_adapter import PostgresCustomChatMessageHistory
from ai_engine.schemas.chat_schemas import ChatInput

# 1. 构建主交通骨干网
master_chain: Runnable = (
        RunnablePassthrough.assign(intent=intent_router)
        | RunnableLambda(route_logger)
        | RunnableLambda(route_logic)
)


# 2. 装配企业级数据库记忆组件
def get_session_history(
        session_id: str,
        tenant_id: str,
        user_id: str,
        lang: str,
) -> BaseChatMessageHistory:
    try:
        _ = lang
        uuid.UUID(session_id)
        valid_session_id = session_id
    except ValueError:
        logger.warning(f"接收到非法的 session_id: {session_id}，已自动替换为新 UUID")
        valid_session_id = str(uuid.uuid4())

    return PostgresCustomChatMessageHistory(
        session_id=valid_session_id,
        tenant_id=tenant_id,
        user_id=user_id
    )


# 3. 导出成品：带记忆、带路由、强类型的企业级对话链
chat_chain = RunnableWithMessageHistory(
    master_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the current chat session",
            default="",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="tenant_id",
            annotation=str,
            name="Tenant ID",
            description="Identifier for the organization or tenant",
            default="default",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the end user",
            default="anonymous",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="lang",
            annotation=str,
            name="Language",
            description="I18n language code for knowledge retrieval (e.g., zh, en, cht)",
            default="zh",
            is_shared=True
        ),
    ]
).with_types(input_type=ChatInput)
