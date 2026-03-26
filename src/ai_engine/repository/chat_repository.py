# src/ai_engine/repository/chat_reposition.py
import uuid
from typing import List, Optional

from sqlalchemy import update, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, asc, col

from ai_engine.models.chat_message import ChatMessage
from ai_engine.models.chat_session import ChatSession


class ChatRepository:
    """
    聊天仓储类：集中处理 Session 和 Message 的数据库增删改查
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==========================================
    # Session (会话) 相关操作
    # ==========================================
    async def get_session(self, session_id: uuid.UUID) -> Optional[ChatSession]:
        """根据 ID 获取会话记录"""
        return await self.db.get(ChatSession, session_id)

    async def get_or_create_session(
            self,
            session_id: uuid.UUID,
            tenant_id: str,
            user_id: str,
            biz_type: Optional[str] = None
    ) -> ChatSession:
        """
        获取会话，如果不存在则自动创建 (Get or Create)。
        这是大模型应用极其常用的逻辑：用户第一次发消息时，自动为他建一个 Session。
        """
        session = await self.get_session(session_id)
        if not session:
            session = ChatSession(
                id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                biz_type=biz_type,
                title="新对话"
            )
            self.db.add(session)
            await self.db.flush()  # flush 会把 SQL 发给数据库并拿到默认值，但还未 commit
        return session

    # ==========================================
    # Message (消息) 相关操作
    # ==========================================
    async def add_message(
            self,
            session_id: uuid.UUID,
            tenant_id: str,
            user_id: str,
            role: str,
            content: str,
            name: Optional[str] = None,
            extra: Optional[dict] = None
    ) -> ChatMessage:
        """添加一条聊天记录"""
        msg = ChatMessage(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            content=content,
            name=name,
            extra=extra or {}
        )
        self.db.add(msg)
        stmt = (
            update(ChatSession)
            .where(col(ChatSession.id) == session_id)
            .values(updated_at=func.now())
        )
        await self.db.execute(stmt)
        await self.db.flush()
        return msg

    async def update_session_title(self, session_id: uuid.UUID, new_title: str) -> None:
        """更新会话的标题"""
        stmt = (
            update(ChatSession)
            .where(col(ChatSession.id) == session_id)
            .values(title=new_title)
        )
        await self.db.execute(stmt)
        await self.db.flush()

    async def get_session_messages(self, session_id: uuid.UUID, limit: int = 50) -> List[ChatMessage]:
        """
        获取某个会话的历史消息（自动过滤软删除，并按时间线正序排列）
        """
        stmt = (
            select(ChatMessage)
            .where(
                col(ChatMessage.session_id) == session_id,
                col(ChatMessage.is_deleted) == False
            )
            .order_by(asc(ChatMessage.created_at))  # 按创建时间正序排，保证对话上下文不出错
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_user_sessions(
            self,
            user_id: str,
            tenant_id: str = "default_tenant",
            limit: int = 20,
            offset: int = 0
    ) -> List[ChatSession]:
        """
        分页获取用户的历史会话列表，按最后更新时间倒序排列。
        过滤掉已逻辑删除的会话。
        """
        stmt = (
            select(ChatSession)
            .where(
                col(ChatSession.user_id) == user_id,
                col(ChatSession.tenant_id) == tenant_id,
                col(ChatSession.is_deleted) == False  # 屏蔽已删除的会话
            )
            .order_by(desc(col(ChatSession.updated_at)))  # 最近聊过的排在最上面
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete_session(self, session_id: uuid.UUID) -> None:
        """
        逻辑删除整个会话（ChatSession）
        """
        stmt = (
            update(ChatSession)
            .where(col(ChatSession.id) == session_id)
            .values(is_deleted=True)
        )
        await self.db.execute(stmt)
        await self.clear_session_messages(session_id)
        await self.db.flush()

    async def clear_session_messages(self, session_id: uuid.UUID) -> None:
        """
        逻辑删除某个会话下的所有消息
        """
        stmt = (
            update(ChatMessage)
            .where(col(ChatMessage.session_id) == session_id)
            .values(is_deleted=True)
        )
        await self.db.execute(stmt)
        await self.db.flush()
