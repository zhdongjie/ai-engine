# src/ai_engine/infra/llm/message_adapter.py
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

# 引入全局单例的数据库连接管理器
from ai_engine.infra.db.pgsql import db_manager
# 引入我们刚才写好的仓储层
from ai_engine.repository.chat_repository import ChatRepository


class PostgresAsyncChatMessageHistory(BaseChatMessageHistory):
    """
    企业级纯异步 LangChain 记忆适配器：
    将 LangChain 的内存消息对象与 PostgreSQL 数据库无缝双向绑定。
    """

    def __init__(
            self,
            session_id: str,
            tenant_id: str = "default_tenant",
            user_id: str = "anonymous",
    ):
        # 确保传入的 session_id 是合法的 UUID
        self.session_id = uuid.UUID(session_id)
        self.tenant_id = tenant_id
        self.user_id = user_id

    # ==========================================
    # 🚫 禁用同步方法，强制纯异步，保护事件循环
    # ==========================================
    @property
    def messages(self) -> List[BaseMessage]:
        raise NotImplementedError(
            "高并发架构下严禁使用同步数据库查询，请使用 await history.aget_messages()"
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        raise NotImplementedError(
            "高并发架构下严禁使用同步数据库写入，请使用 await history.aadd_messages()"
        )

    def clear(self) -> None:
        raise NotImplementedError(
            "高并发架构下严禁使用同步数据库操作，请使用 await history.aclear()"
        )

    # ==========================================
    # ⚡ 核心异步实现 (被 LangChain 的 ainvoke/astream 自动调用)
    # ==========================================
    async def aget_messages(self) -> List[BaseMessage]:
        """
        从 PostgreSQL 读取当前会话的历史记录，
        并将其翻译成 LangChain 认识的 Message 对象。
        """
        # 临时从连接池借用一个连接，离开 with 块时自动归还
        async with db_manager.session_context() as db:
            repo = ChatRepository(db)
            db_messages = await repo.get_session_messages(self.session_id)

            lc_messages = []
            for msg in db_messages:
                if msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))
                elif msg.role == "tool":
                    lc_messages.append(
                        ToolMessage(content=msg.content, tool_call_id=msg.name or "unknown_tool")
                    )

            return lc_messages

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        接收大模型（或用户）生成的新消息，
        将其翻译成数据库实体，并持久化到 PostgreSQL。
        """
        async with db_manager.session_context() as db:
            repo = ChatRepository(db)

            # 1. 确保当前会话存在 (如果不存在则自动创建，实现前后端无感解耦)
            # 这里默认打上业务标识，方便后续做路由和数据隔离
            await repo.get_or_create_session(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                biz_type="power_platform"
            )

            # 2. 遍历 LangChain 传过来的消息，并逐一翻译
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
                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "tool"
                    name = msg.tool_call_id  # 提取具体的工具调用标识

                # 将内容强制转为字符串，防止 LangChain 传递复杂对象结构报错
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)

                # 写入仓储层 (这里只做 flush，等待循环结束后统一 commit)
                await repo.add_message(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    role=role,
                    content=content_str,
                    name=name,
                    extra=extra
                )

            # 3. 统一提交事务，保证多条消息（如思考过程 + 最终回答）的原子性写入
            await db.commit()

    async def aclear(self) -> None:
        """
        清空当前会话的记忆。
        (考虑到合规与数据沉淀，企业级应用通常不推荐物理删除，可以先留空，或后续在 repo 实现软删除)
        """
        pass
