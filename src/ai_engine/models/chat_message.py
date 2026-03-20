# src/ai_engine/models/chat_message.py
from typing import Dict, Any, Optional, TYPE_CHECKING
from uuid import UUID

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, Column

from ai_engine.infra.db.base_model import BaseModel
from ai_engine.infra.db.mixins import (
    TimestampMixin,
    TenantMixin,
    SoftDeleteMixin
)

if TYPE_CHECKING:
    from ai_engine.models.chat_session import ChatSession


class ChatMessage(
    BaseModel,
    TimestampMixin,
    TenantMixin,
    SoftDeleteMixin,
    table=True
):
    """
    企业级 AI 消息流水表
    """
    __tablename__ = "chat_messages"

    # 核心关联
    session_id: UUID = Field(foreign_key="chat_sessions.id", index=True)
    user_id: str = Field(index=True, description="冗余的用户ID，极大提升跨会话的 Token 计费与聚合查询性能")

    # 消息正文与角色
    role: str = Field(max_length=20, description="角色: system, user, assistant, tool")
    name: Optional[str] = Field(default=None, max_length=64, description="当 role=tool 时，记录具体调用的工具名称")
    content: str = Field(default="", description="消息文本内容 (或 Agent 的思考过程)")

    # 对话结构控制
    parent_id: Optional[UUID] = Field(
        default=None,
        foreign_key="chat_messages.id",
        index=True,
        description="父消息ID，用于支持像 ChatGPT 一样的『重新生成(Regenerate)』形成的分支树形结构"
    )

    # 竞态条件说明：如果你在应用层没有做严格的分布式锁，建议不要强依赖手动计算的 position，
    # 而是依赖 parent_id 形成的链表，或者底层的 created_at 时间戳进行排序。
    position: int = Field(default=0, index=True, description="消息在当前分支下的序号")

    # 异步与流式状态
    status: str = Field(
        default="completed",
        max_length=20,
        description="状态: pending(排队中), streaming(流式输出中), completed(完成), error(异常)"
    )

    # 高级元数据存储
    # 建议在此处存储：
    # 1. token_usage: {"prompt_tokens": 10, "completion_tokens": 20}
    # 2. tool_calls: OpenAI 格式的工具调用请求 JSON
    # 3. citations: RAG 检索命中的文档引用来源 (如 virtual_card 的 markdown 切片)
    # 4. error_msg: 如果 status=error，这里记录详细报错堆栈
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, comment="高度灵活的元数据(Token, 工具调用, RAG引用)")
    )

    # 关联会话与自关联
    session: "ChatSession" = Relationship(back_populates="messages")
