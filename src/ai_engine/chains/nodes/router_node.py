# src/ai_engine/chains/nodes/router_node.py
from typing import Any, Dict, AsyncIterator, List

from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableConfig

from ai_engine.chains.nodes.normal_node import normal_chat_run
from ai_engine.chains.nodes.rag_node import dynamic_rag_run
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.llm_factory import LLMFactory
from ai_engine.schemas.chat_schemas import ChatInput, ChatOutput

# =========================
# 1. Intent Router
# =========================
router_config = get_prompt_config("intent_router")

router_llm = LLMFactory.get_model(router_config.get("config", {}))
router_prompt = PromptTemplate.from_template(router_config["content"])

intent_router = router_prompt | router_llm | StrOutputParser()


# =========================
# 2. i18n 拦截器
# =========================
def i18n_input_interceptor(info: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    在用户输入前加语言控制指令（最简单稳定方案）
    """
    configurable = config.get("configurable") or {}
    user_lang = configurable.get("lang", "zh")

    input_patches = {
        "en": "[System Instruction: Please respond in English] ",
        "cht": "[系統指令：請務必使用繁體中文回答] ",
    }

    patch = input_patches.get(user_lang)

    if patch:
        original_input = info.get("input", "")
        info["input"] = f"{patch}{original_input}"

    return info


# =========================
# 3. 日志节点
# =========================
def route_logger(info: Dict[str, Any]) -> Dict[str, Any]:
    intent = info.get("intent", "UNKNOWN").strip().upper()
    logger.info(f"[Router] intent={intent} | input={info.get('input', '')[:30]}")
    return info


# =========================
# 4. metadata 合并函数
# =========================
def merge_metadata(config: RunnableConfig, new_data: Dict[str, Any]) -> Dict[str, Any]:
    old = config.get("metadata") or {}
    return {**old, **new_data}


# =========================
# 5. Stream Formatter
# =========================
async def response_stream_formatter(
        input_stream: AsyncIterator[Any],
        config: RunnableConfig
) -> AsyncIterator[AIMessageChunk]:
    metadata = config.get("metadata") or {}

    intent = metadata.get("intent", "NORMAL")
    biz_type = metadata.get("biz_type", "normal_chat")

    sources: List[Any] = []

    async for chunk in input_stream:

        # -------- 统一兼容 chunk --------
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            additional_kwargs = chunk.get("additional_kwargs", {})
            response_metadata = chunk.get("response_metadata", {})
            chunk_id = chunk.get("id")
        else:
            content = getattr(chunk, "content", "")
            additional_kwargs = getattr(chunk, "additional_kwargs", {})
            response_metadata = getattr(chunk, "response_metadata", {})
            chunk_id = getattr(chunk, "id", None)

        # -------- 输出内容 --------
        if content:
            yield AIMessageChunk(**{
                "content": content,
                "additional_kwargs": additional_kwargs,
                "response_metadata": response_metadata,
                "id": chunk_id
            })

        # -------- 累加 sources --------
        if additional_kwargs and "sources" in additional_kwargs:
            new_sources = additional_kwargs.get("sources") or []
            sources.extend(new_sources)

    # -------- 去重 --------
    sources = list({str(s): s for s in sources}.values())

    # -------- 最终收尾 chunk --------
    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "sources": sources,
            "intent": intent,
            "biz_type": biz_type
        }
    })


# =========================
# 6. 核心路由逻辑
# =========================
def route_logic(info: Dict[str, Any]):
    intent_str = (info.get("intent") or "").strip().upper()

    # -------- fallback（可选：用 LLM 判 intent）--------
    if not intent_str:
        try:
            intent_str = intent_router.invoke({
                "input": info.get("input", "")
            }).strip().upper()
        except Exception as e:
            logger.warning(f"Intent Router fallback failed: {e}")
            intent_str = "NORMAL"

    # -------- 选节点 --------
    if intent_str == "RAG":
        core_node = RunnableLambda(dynamic_rag_run)
    else:
        core_node = RunnableLambda(normal_chat_run)

    # -------- 构建 chain --------
    chain = (
            RunnableLambda(i18n_input_interceptor)
            | RunnableLambda(route_logger)
            | core_node
            | response_stream_formatter # type: ignore
    )

    return chain.with_types(
        input_type=ChatInput,
        output_type=ChatOutput
    )
