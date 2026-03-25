# src/ai_engine/api/chat_router.py
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

# 创建路由器
router = APIRouter(prefix="/chat/sessions", tags=["Session Management"])


class SessionItem(BaseModel):
    # ... 定义返回结构 ...
    pass


@router.get("", response_model=List[SessionItem])
async def get_chat_sessions():
    # ... 获取列表逻辑 ...
    pass


@router.delete("/{session_id}")
async def delete_chat_session():
    # ... 删除会话逻辑 ...
    pass


@router.delete("/{session_id}/clear")
async def clear_session_memory():
    # ... 清空聊天记录逻辑 ...
    pass
