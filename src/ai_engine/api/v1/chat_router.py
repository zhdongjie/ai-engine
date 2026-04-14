# src/ai_engine/api/chat_router.py
import uuid
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ai_engine.infra.db.pgsql import db_manager
from ai_engine.repository.chat_repository import AsyncChatRepository
from ai_engine.schemas.chat_schemas import SessionItem
from ai_engine.schemas.result import Result
from ai_engine.core.constants import ResponseCode

# 创建路由器
router = APIRouter(prefix="/chat/sessions", tags=["Session Management"])

# 统一的响应描述模板，减少重复代码
COMMON_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    401: {"model": Result, "description": "未授权：Header 缺失或无效"},
    404: {"model": Result, "description": "未找到：资源不存在"},
    500: {"model": Result, "description": "系统异常：服务器内部错误"}
}

async def get_user_id(x_user_id: str = Header(..., alias="X-User-Id")):
    """
    依赖注入：获取网关透传的用户ID
    """
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id"
        )
    return x_user_id

@router.get(
    "",
    response_model=Result[List[SessionItem]],
    summary="获取会话列表",
    description="获取当前登录用户的所有历史会话记录，按最后更新时间倒序排列。支持逻辑删除过滤。",
    responses={200: {"description": "成功返回会话列表"}, **COMMON_RESPONSES}
)
async def get_chat_sessions(
        user_id: str = Depends(get_user_id),
        db: AsyncSession = Depends(db_manager.get_async_session)
):
    repo = AsyncChatRepository(db)
    # 调用 repository 分页获取未被逻辑删除的会话
    sessions = await repo.get_user_sessions(user_id=user_id)
    return Result.success(data=sessions)


@router.delete(
    "/{session_id}",
    response_model=Result,
    summary="删除特定会话",
    description="执行会话的逻辑删除。删除后该会话及其消息将不再出现在列表中。",
    responses={200: {"description": "会话成功标记为已删除"}, **COMMON_RESPONSES}
)
async def delete_chat_session(
        session_id: uuid.UUID,
        user_id: str = Depends(get_user_id),
        db: AsyncSession = Depends(db_manager.get_async_session)
):
    repo = AsyncChatRepository(db)
    # 鉴权：检查会话归属
    session = await repo.get_session(session_id)
    if not session or session.user_id != user_id:
        return Result.fail(code=ResponseCode.NOT_FOUND.value, msg="会话不存在")

    await repo.delete_session(session_id)
    return Result.success(msg="会话已成功删除")


@router.delete(
    "/{session_id}/clear",
    response_model=Result,
    summary="清空会话聊天记录",
    description="保留会话 ID 和配置，但逻辑删除该会话下的所有历史消息。常用于重置对话状态。",
    responses={200: {"description": "消息记忆已清空"}, **COMMON_RESPONSES}
)
async def clear_session_memory(
        session_id: uuid.UUID,
        user_id: str = Depends(get_user_id),
        db: AsyncSession = Depends(db_manager.get_async_session)
):
    repo = AsyncChatRepository(db)
    # 鉴权：检查会话归属
    session = await repo.get_session(session_id)
    if not session or session.user_id != user_id:
        return Result.fail(code=ResponseCode.NOT_FOUND.value, msg="会话不存在")

    await repo.clear_session_messages(session_id)
    return Result.success(msg="记忆已成功清空")