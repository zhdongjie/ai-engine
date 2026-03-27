# src/ai_engine/schemas/chat_schemas.py
from pydantic import BaseModel, Field


class ChatInput(BaseModel):
    input: str = Field(..., description="用户的纯文本提问")
    biz_type: str = Field(default="normal_chat", description="业务类型标识符")
