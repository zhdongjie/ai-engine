# src/ai_engine/schemas/chat_schemas.py
from datetime import datetime
from typing import Optional, List

from pydantic import Field

from ai_engine.schemas.base import BaseSchema


class ChatInput(BaseSchema):
    input: str = Field(..., description="用户的纯文本提问")
    biz_type: str = Field(default="normal_chat", description="业务类型标识符")


class ChatOutput(BaseSchema):
    """
    全系统统一的结构化输出模型
    """
    answer: str = Field(..., description="AI 生成的核心回答内容")
    sources: List[str] = Field(default_factory=list, description="参考来源文档列表")
    intent: str = Field(default="NORMAL", description="意图识别结果")


class SessionItem(BaseSchema):
    """会话列表项，用于前端展示"""
    id: str = Field(..., description="会话唯一 ID")
    title: str = Field(..., description="会话标题")
    biz_type: Optional[str] = Field(None, description="业务类型场景")
    updated_at: datetime = Field(..., description="最后活跃时间")
    is_pinned: bool = Field(default=False, description="是否置顶")
