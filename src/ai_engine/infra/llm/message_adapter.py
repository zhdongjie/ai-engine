import threading
import uuid
from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.title_chain import generate_session_title
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.repository.chat_repository import ChatRepository


def _background_generate_title(session_id: uuid.UUID, user_content: str, lang: str):
    """
    后台独立线程任务：生成标题并更新数据库
    @param lang: 传入语言上下文，确保标题生成符合用户语言偏好
    """
    logger.info(f"开始为会话 {session_id} 生成标题 (语言: {lang})...")

    try:
        config: RunnableConfig = {"configurable": {"lang": lang}}
        new_title = generate_session_title(user_content, config=config)

        with db_manager.session_context() as db:
            repo = ChatRepository(db)
            repo.update_session_title(session_id, new_title)
            db.commit()
            logger.success(f"会话 {session_id} 标题更新成功: 【{new_title}】")
    except Exception as e:
        logger.error(f"后台生成标题失败 | Session: {session_id} | Error: {e}")


class PostgresCustomChatMessageHistory(BaseChatMessageHistory):
    """
    企业级 PostgreSQL 消息适配器：
    支持元数据（模型、人设、Token、来源）的自动同步与持久化。
    """

    def __init__(
            self,
            session_id: str,
            tenant_id: str = "default",
            user_id: str = "anonymous",
    ):
        self.session_id = uuid.UUID(session_id)
        self.tenant_id = tenant_id
        self.user_id = user_id

    @property
    def messages(self) -> List[BaseMessage]:
        """读取历史记录：将数据库行还原为 LangChain 消息对象"""
        with db_manager.session_context() as db:
            repo = ChatRepository(db)
            db_messages = repo.get_session_messages(self.session_id)

            lc_messages = []
            for msg in db_messages:
                if msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content, additional_kwargs=msg.extra))
                elif msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))
                elif msg.role == "tool":
                    lc_messages.append(ToolMessage(content=msg.content, tool_call_id=msg.name or "unknown_tool"))

            # 历史消息截断，防止 Token 溢出
            if len(lc_messages) > settings.MAX_HISTORY_MESSAGES:
                return lc_messages[-settings.MAX_HISTORY_MESSAGES:]

            return lc_messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """同步写入新消息并自动补全 Session 元数据"""
        with db_manager.session_context() as db:
            repo = ChatRepository(db)

            # --- 1. 预扫描元数据 ---
            biz_type = "default"
            current_lang = "zh"
            current_model = None
            current_system_prompt = None

            for message in messages:
                if isinstance(message, AIMessage) and message.additional_kwargs:
                    kw = message.additional_kwargs
                    biz_type = kw.get("biz_type", biz_type)
                    current_lang = kw.get("lang", current_lang)
                    current_model = kw.get("model_name", current_model)
                    current_system_prompt = kw.get("system_prompt", current_system_prompt)

            # --- 2. 同步 Session 状态 ---
            session = repo.get_or_create_session(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                biz_type=biz_type
            )

            # 自动补全 Session 字段
            if not getattr(session, "model_name", None) and current_model:
                session.model_name = current_model
            if not getattr(session, "system_prompt", None) and current_system_prompt:
                session.system_prompt = current_system_prompt

            if hasattr(session, "lang"):
                session.lang = current_lang

            # --- 3. 消息入库 ---
            for msg in messages:
                role = "user"
                name = None
                extra = {}

                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    # 整合来自 Runner 的业务元数据
                    if msg.additional_kwargs:
                        extra = {k: v for k, v in msg.additional_kwargs.items() if k != "injected_messages"}

                    # 整合来自 Provider 的响应元数据
                    if msg.response_metadata:
                        extra["model_name"] = msg.response_metadata.get("model_name", extra.get("model_name"))
                        extra["model_provider"] = msg.response_metadata.get("model_provider",
                                                                            extra.get("model_provider"))

                    # Token 统计归一化
                    usage = getattr(msg, "usage_metadata", None) or extra.get("usage_metadata")
                    if usage:
                        extra["token_usage"] = usage

                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "tool"
                    name = msg.tool_call_id

                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                repo.add_message(
                    session_id=self.session_id, tenant_id=self.tenant_id,
                    user_id=self.user_id, role=role, content=content_str,
                    name=name, extra=extra
                )

            db.commit()

        # --- 4. 异步触发标题生成 ---
        if getattr(session, "title", "") == "新对话":
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            if human_msgs:
                # 开启守护线程，避免阻塞主流输出
                threading.Thread(
                    target=_background_generate_title,
                    args=(self.session_id, human_msgs[0].content, current_lang),
                    daemon=True
                ).start()

    def clear(self) -> None:
        """清空会话"""
        with db_manager.session_context() as db:
            repo = ChatRepository(db)
            repo.clear_session_messages(self.session_id)
            db.commit()
