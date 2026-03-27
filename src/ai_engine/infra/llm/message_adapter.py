# src/ai_engine/infra/llm/message_adapter.py
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

from ai_engine.chains.title_chain import generate_session_title  # 假设你也把它改成同步了
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.repository.chat_repository import ChatRepository


def _background_generate_title(session_id: uuid.UUID, user_content: str):
    """后台独立线程任务：生成标题并更新数据库"""
    logger.info(f"正在后台为会话 {session_id} 自动生成标题...")

    try:
        new_title = generate_session_title(user_content)

        with db_manager.session_context() as db:
            repo = ChatRepository(db)
            repo.update_session_title(session_id, new_title)
            db.commit()
            logger.success(f"会话 {session_id} 标题已成功更新为: 【{new_title}】")
    except Exception as e:
        logger.error(f"后台生成标题失败: {e}")


class PostgresCustomChatMessageHistory(BaseChatMessageHistory):
    """
    企业级同步 LangChain 记忆适配器：
    将 LangChain 的内存消息对象与 PostgreSQL 数据库无缝双向绑定。
    支持 Token 聚合统计与元数据持久化。
    """

    def __init__(
            self,
            session_id: str,
            tenant_id: str = "default_tenant",
            user_id: str = "anonymous",
    ):
        self.session_id = uuid.UUID(session_id)
        self.tenant_id = tenant_id
        self.user_id = user_id

    @property
    def messages(self) -> List[BaseMessage]:
        """LangChain 要求的同步属性：读取历史记录"""
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

            if len(lc_messages) > settings.MAX_HISTORY_MESSAGES:
                truncated_messages = lc_messages[-settings.MAX_HISTORY_MESSAGES:]
                logger.debug(f"会话 {self.session_id} 历史超限，截断至 {settings.MAX_HISTORY_MESSAGES} 条。")
                return truncated_messages

            return lc_messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """LangChain 要求的同步方法：写入新消息"""
        with db_manager.session_context() as db:
            repo = ChatRepository(db)

            biz_type = "default"
            for m in messages:
                if isinstance(m, AIMessage) and m.additional_kwargs:
                    biz_type = m.additional_kwargs.get("biz_type", biz_type)

            session = repo.get_or_create_session(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                biz_type=biz_type
            )

            for msg in messages:
                role = "user"
                name = None
                extra = {}

                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    if msg.additional_kwargs:
                        extra.update(msg.additional_kwargs)
                    if msg.response_metadata:
                        extra.update(msg.response_metadata)

                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        extra["token_usage"] = usage
                    elif "token_usage" in msg.response_metadata:
                        extra["token_usage"] = msg.response_metadata["token_usage"]

                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "tool"
                    name = msg.tool_call_id

                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)

                repo.add_message(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    role=role,
                    content=content_str,
                    name=name,
                    extra=extra
                )

            db.commit()

        # 使用 Python 原生线程替代 asyncio.create_task 处理后台任务
        if getattr(session, "title", "") == "新对话":
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            if human_msgs:
                threading.Thread(
                    target=_background_generate_title,
                    args=(self.session_id, human_msgs[0].content),
                    daemon=True
                ).start()

    def clear(self) -> None:
        """清空会话 (逻辑删除)"""
        with db_manager.session_context() as db:
            repo = ChatRepository(db)
            repo.clear_session_messages(self.session_id)
            db.commit()
