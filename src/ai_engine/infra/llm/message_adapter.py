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
# 引入仓储层处理数据库增删改查
from ai_engine.repository.chat_repository import ChatRepository


class PostgresAsyncChatMessageHistory(BaseChatMessageHistory):
    """
    企业级纯异步 LangChain 记忆适配器：
    将 LangChain 的内存消息对象与 PostgreSQL 数据库无缝双向绑定。
    支持流式输出后的 Token 聚合统计与元数据持久化。
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
        raise NotImplementedError("高并发架构下严禁使用同步查询，请使用 await history.aget_messages()")

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        raise NotImplementedError("高并发架构下严禁使用同步写入，请使用 await history.aadd_messages()")

    def clear(self) -> None:
        raise NotImplementedError("高并发架构下严禁使用同步操作，请使用 await history.aclear()")

    # ==========================================
    # ⚡ 核心异步实现 (被 LangChain 自动调用)
    # ==========================================
    async def aget_messages(self) -> List[BaseMessage]:
        """从 PostgreSQL 读取当前会话的历史记录"""
        async with db_manager.session_context() as db:
            repo = ChatRepository(db)
            db_messages = await repo.get_session_messages(self.session_id)

            lc_messages = []
            for msg in db_messages:
                if msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    # 将存储在 extra 中的元数据还原到消息对象中
                    lc_messages.append(AIMessage(content=msg.content, additional_kwargs=msg.extra))
                elif msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))
                elif msg.role == "tool":
                    lc_messages.append(
                        ToolMessage(content=msg.content, tool_call_id=msg.name or "unknown_tool")
                    )
            return lc_messages

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        持久化新消息。
        优化点：捕获流式聚合后的 usage_metadata (Token) 和手动注入的业务标识。
        """
        async with db_manager.session_context() as db:
            repo = ChatRepository(db)

            # 1. 动态确定业务类型 (从 AI 消息中提取我们在 chat_chain 注入的标识)
            biz_type = "default"
            for m in messages:
                if isinstance(m, AIMessage) and m.additional_kwargs:
                    biz_type = m.additional_kwargs.get("biz_type", biz_type)

            # 2. 确保当前会话存在
            await repo.get_or_create_session(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                biz_type=biz_type
            )

            # 3. 遍历 LangChain 传过来的消息，并逐一翻译
            for msg in messages:
                role = "user"
                name = None
                extra = {}

                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    # A. 提取手动注入的元数据 (如 sources, biz_type, has_context)
                    if msg.additional_kwargs:
                        extra.update(msg.additional_kwargs)

                    # B. 提取 LLM 响应元数据 (如 model_name)
                    if msg.response_metadata:
                        extra.update(msg.response_metadata)

                    # C. 关键：提取流式聚合后的标准 Token 统计数据 (LangChain 0.2+ 推荐方式)
                    # 使用 getattr 规避 IDE 对动态属性的类型警告
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        extra["token_usage"] = usage
                    # 备选方案：如果 usage_metadata 为空，尝试从 response_metadata 寻找
                    elif "token_usage" in msg.response_metadata:
                        extra["token_usage"] = msg.response_metadata["token_usage"]

                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "tool"
                    name = msg.tool_call_id

                # 确保内容为字符串
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)

                # 写入持久化层
                await repo.add_message(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    role=role,
                    content=content_str,
                    name=name,
                    extra=extra
                )

            # 统一提交事务
            await db.commit()

    async def aclear(self) -> None:
        """清空会话（通常建议实现为软删除）"""
        pass