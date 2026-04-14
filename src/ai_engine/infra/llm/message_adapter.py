import asyncio
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
from langchain_core.runnables import RunnableConfig

from ai_engine.chains.title_chain import generate_session_title
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.repository.chat_repository import AsyncChatRepository

DEFAULT_SESSION_TITLE = "新对话"


async def _background_generate_title(session_id: uuid.UUID, user_content: str, lang: str) -> None:
    """Generate and persist a session title in the background."""
    logger.info(f"????? {session_id} ???? (??: {lang})...")

    try:
        config: RunnableConfig = {"configurable": {"lang": lang}}
        new_title = await generate_session_title(user_content, config=config)

        async with db_manager.async_session_context() as db:
            repo = AsyncChatRepository(db)
            await repo.update_session_title(session_id, new_title)
            await db.commit()
            logger.success(f"?? {session_id} ??????: ?{new_title}?")
    except Exception as e:
        logger.error(f"???????? | Session: {session_id} | Error: {e}")


class PostgresCustomChatMessageHistory(BaseChatMessageHistory):
    """Async Postgres-backed chat history with metadata persistence."""

    def __init__(
            self,
            session_id: str,
            tenant_id: str = "default",
            user_id: str = "anonymous",
    ):
        self.session_id = uuid.UUID(session_id)
        self.tenant_id = tenant_id
        self.user_id = user_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Sync fallback for message history (use aget_messages in async flows)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aget_messages())
        raise RuntimeError("Use aget_messages() in async context.")

    async def aget_messages(self) -> List[BaseMessage]:
        """Async fetch of message history."""
        async with db_manager.async_session_context() as db:
            repo = AsyncChatRepository(db)
            db_messages = await repo.get_session_messages(self.session_id)

            lc_messages: List[BaseMessage] = []
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
                return lc_messages[-settings.MAX_HISTORY_MESSAGES:]

            return lc_messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Sync fallback for adding messages (use aadd_messages in async flows)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._aadd_messages(messages, background=False))
            return
        raise RuntimeError("Use aadd_messages() in async context.")

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Async add messages with metadata persistence."""
        await self._aadd_messages(messages, background=True)

    async def _aadd_messages(self, messages: Sequence[BaseMessage], background: bool) -> None:
        async with db_manager.async_session_context() as db:
            repo = AsyncChatRepository(db)

            # --- 1. Extract metadata ---
            biz_type = "default"
            current_lang = "zh"
            current_model = None
            current_system_prompt = None

            for message in messages:
                if isinstance(message, AIMessage) and message.additional_kwargs:
                    kw = message.additional_kwargs
                    biz_type = kw.get("biz_type", biz_type)
                    current_lang = kw.get("lang", current_lang)
                    current_model = kw.get("model_name", current_model)
                    current_system_prompt = kw.get("system_prompt", current_system_prompt)

            # --- 2. Ensure session ---
            session = await repo.get_or_create_session(
                session_id=self.session_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                biz_type=biz_type,
            )

            if not getattr(session, "model_name", None) and current_model:
                session.model_name = current_model
            if not getattr(session, "system_prompt", None) and current_system_prompt:
                session.system_prompt = current_system_prompt
            if hasattr(session, "lang"):
                session.lang = current_lang

            # --- 3. Persist messages ---
            for msg in messages:
                role = "user"
                name = None
                extra = {}

                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                    if msg.additional_kwargs:
                        extra = {k: v for k, v in msg.additional_kwargs.items() if k != "injected_messages"}

                    if msg.response_metadata:
                        extra["model_name"] = msg.response_metadata.get("model_name", extra.get("model_name"))
                        extra["model_provider"] = msg.response_metadata.get("model_provider", extra.get("model_provider"))

                    usage = getattr(msg, "usage_metadata", None) or extra.get("usage_metadata")
                    if usage:
                        extra["token_usage"] = usage

                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, ToolMessage):
                    role = "tool"
                    name = msg.tool_call_id

                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                await repo.add_message(
                    session_id=self.session_id,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    role=role,
                    content=content_str,
                    name=name,
                    extra=extra,
                )

            await db.commit()
            session_title = getattr(session, "title", "")

        # --- 4. Trigger title generation ---
        if session_title == DEFAULT_SESSION_TITLE:
            human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            if human_msgs:
                if background:
                    asyncio.create_task(
                        _background_generate_title(self.session_id, human_msgs[0].content, current_lang)
                    )
                else:
                    await _background_generate_title(self.session_id, human_msgs[0].content, current_lang)

    def clear(self) -> None:
        """Sync fallback for clearing messages (use aclear in async flows)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclear())
            return
        raise RuntimeError("Use aclear() in async context.")

    async def aclear(self) -> None:
        """Async clear all messages."""
        async with db_manager.async_session_context() as db:
            repo = AsyncChatRepository(db)
            await repo.clear_session_messages(self.session_id)
            await db.commit()
