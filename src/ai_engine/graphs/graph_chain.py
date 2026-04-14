import uuid
from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import ConfigurableFieldSpec, Runnable, RunnableConfig, ensure_config
from ai_engine.graphs.state import ChatGraphState
from ai_engine.schemas.chat_schemas import ChatInput
from ai_engine.graphs.workflow import get_graph_app


class GraphChatChain(Runnable[ChatGraphState, AIMessageChunk]):
    """Runnable wrapper that exposes the LangGraph workflow to LangServe."""

    def __init__(self) -> None:
        super().__init__()
        self._config_specs = [
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the current chat session",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="tenant_id",
                annotation=str,
                name="Tenant ID",
                description="Identifier for the organization or tenant",
                default="default",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the end user",
                default="anonymous",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="lang",
                annotation=str,
                name="Language",
                description="I18n language code for knowledge retrieval (e.g., zh, en, cht)",
                default="zh",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="user_level",
                annotation=str,
                name="User Level",
                description="User expertise level for selecting specific prompts (e.g., simple, expert, default)",
                default="default",
                is_shared=True,
            ),
        ]

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self._config_specs

    def invoke(self, input: ChatGraphState, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessageChunk:
        raise RuntimeError("GraphChatChain only supports async execution (use ainvoke/astream).")

    async def ainvoke(self, input: ChatGraphState, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessageChunk:
        app = await get_graph_app()
        normalized = ensure_config(config)
        _ensure_thread_id(normalized)
        output = await app.ainvoke(input, config=normalized, **kwargs)
        if isinstance(output, dict):
            return _final_chunk_from_state(output)
        return AIMessageChunk(content=str(output))

    async def astream(
        self,
        input: ChatGraphState,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        app = await get_graph_app()
        normalized = ensure_config(config)
        _ensure_thread_id(normalized)
        normalized["callbacks"] = normalized.get("callbacks") or []

        async for chunk in app.astream(
            input,
            config=normalized,
            stream_mode="custom",
            **kwargs,
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "llm_chunk":
                content = chunk.get("content", "")
                yield AIMessageChunk(content=content)
            elif isinstance(chunk, dict) and chunk.get("type") == "final_chunk":
                metadata = chunk.get("metadata")
                if isinstance(metadata, dict):
                    yield AIMessageChunk(content="", additional_kwargs=metadata)
                else:
                    yield AIMessageChunk(content="")
                return

    async def astream_events(
        self,
        input: ChatGraphState,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        app = await get_graph_app()
        normalized = ensure_config(config)
        _ensure_thread_id(normalized)

        run_id = str(uuid.uuid4())
        tags = normalized.get("tags", [])
        metadata = normalized.get("metadata", {})

        yield {
            "event": "on_chain_start",
            "name": "graph_chat",
            "run_id": run_id,
            "tags": tags,
            "metadata": metadata,
            "data": {"input": input},
            "parent_ids": [],
        }

        answer_parts: List[str] = []

        async for chunk in app.astream(
            input,
            config=normalized,
            stream_mode="custom",
            **kwargs,
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "llm_chunk":
                content = chunk.get("content", "")
                if content:
                    answer_parts.append(content)
                    yield {
                        "event": "on_llm_stream",
                        "name": "graph_chat",
                        "run_id": run_id,
                        "tags": tags,
                        "metadata": metadata,
                        "data": {"chunk": AIMessageChunk(content=content)},
                        "parent_ids": [],
                    }
            elif isinstance(chunk, dict) and chunk.get("type") == "final_chunk":
                final_meta = chunk.get("metadata") if isinstance(chunk, dict) else {}
                yield {
                    "event": "on_chain_end",
                    "name": "graph_chat",
                    "run_id": run_id,
                    "tags": tags,
                    "metadata": metadata,
                    "data": {
                        "output": {
                            "final_answer": "".join(answer_parts).strip(),
                            "response_metadata": final_meta or {},
                        }
                    },
                    "parent_ids": [],
                }
                return

        yield {
            "event": "on_chain_end",
            "name": "graph_chat",
            "run_id": run_id,
            "tags": tags,
            "metadata": metadata,
            "data": {"output": {"final_answer": "".join(answer_parts).strip()}},
            "parent_ids": [],
        }


graph_chat_chain = GraphChatChain().with_types(input_type=ChatInput)


def _final_chunk_from_state(state: Dict[str, Any]) -> AIMessageChunk:
    metadata = state.get("response_metadata") or {}
    return AIMessageChunk(
        content="",
        additional_kwargs=metadata if isinstance(metadata, dict) else {},
    )


def _ensure_thread_id(config: RunnableConfig) -> None:
    configurable = config.get("configurable") or {}
    if not configurable.get("thread_id"):
        session_id = configurable.get("session_id")
        if session_id:
            configurable["thread_id"] = session_id
            config["configurable"] = configurable
            return
        raise ValueError("Missing `thread_id` or `session_id` in config.configurable for LangGraph checkpointer")
