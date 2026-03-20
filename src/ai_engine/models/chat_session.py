# src/ai_engine/models/chat_session.py
from typing import List, Optional, TYPE_CHECKING

from sqlmodel import Field, Relationship

from ai_engine.infra.db.base_model import BaseModel
from ai_engine.infra.db.mixins import (
    TimestampMixin,
    TenantMixin,
    SoftDeleteMixin
)

if TYPE_CHECKING:
    from ai_engine.models.chat_message import ChatMessage


class ChatSession(
    BaseModel,
    TimestampMixin,
    TenantMixin,
    SoftDeleteMixin,
    table=True
):
    """
    企业级 AI 会话主表
    """
    __tablename__ = "chat_sessions"

    # 用户与基础信息
    user_id: str = Field(index=True, description="发起会话的用户标识")
    title: Optional[str] = Field(default="新对话", max_length=255, description="会话标题（可通过 LLM 总结）")

    # 业务与路由控制
    biz_type: Optional[str] = Field(
        default=None,
        index=True,
        description="业务场景路由标识 (例如: virtual_card, kyc_process, stock_trading)"
    )
    model_provider: Optional[str] = Field(
        default="openai",
        max_length=50,
        description="默认模型供应商 (如 openai, qwen, zhipu)"
    )
    model_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="当前会话绑定的模型版本 (如 gpt-4o, qwen-plus)"
    )

    # 上下文与状态控制
    system_prompt: Optional[str] = Field(
        default=None,
        description="当前会话注入的 System Prompt，如果不填则使用 biz_type 对应的默认配置"
    )
    summary: Optional[str] = Field(
        default=None,
        description="长会话的记忆摘要（当 Context Window 不足时，由 LLM 自动生成的历史摘要）"
    )
    is_pinned: bool = Field(default=False, description="用户是否将该对话置顶")

    # 关联消息
    messages: List["ChatMessage"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
