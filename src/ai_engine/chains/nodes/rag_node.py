# src/ai_engine/chains/nodes/rag_node.py
import asyncio
from typing import Dict, Any, AsyncIterator

from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings
from ai_engine.infra.db.vdb import vdb_manager
from ai_engine.infra.llm.llm_factory import LLMFactory
from ai_engine.utils.retrieval_utils import get_reranked_docs, format_docs_with_sources


async def dynamic_rag_run(input_data: Dict[str, Any]) -> AsyncIterator[BaseMessage]:
    """异步版本：原子化执行 RAG 检索生成"""
    biz_type = input_data.get("biz_type", "normal_chat")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])

    prompt_data = get_prompt_config(biz_type)

    retrieval_config = prompt_data.get("retrieval_config", {})
    search_k = retrieval_config.get("k", settings.VECTOR_SEARCH_TOP_K)

    logger.debug(f"进入 RAG 模式，开始检索知识库 (召回数量 k={search_k})...")

    retriever = vdb_manager.store.as_retriever(search_kwargs={"k": search_k})
    initial_docs = await asyncio.to_thread(retriever.invoke, user_input)
    logger.info(f"向量库初筛完成，抓取到原始文档: {len(initial_docs)} 篇")

    final_docs = await asyncio.to_thread(get_reranked_docs, user_input, initial_docs)
    logger.info(f"重排阶段完成，剩余精选文档: {len(final_docs)}")
    context, sources = format_docs_with_sources(final_docs)

    if not context.strip():
        logger.warning(f"检索结果为空，触发 RAG 强制静默，已阻断大模型调用。")
        yield AIMessageChunk(**{"content": "抱歉，知识库中未能检索到与您问题相关的信息。请尝试换个说法。"})

        # 给出最终的元数据标记，结束本次流
        yield AIMessageChunk(**{
            "content": "",
            "additional_kwargs": {
                "sources": [],
                "biz_type": biz_type,
                "has_context": False
            }
        })
        return

    if final_docs:
        new_biz_type = final_docs[0].metadata.get("biz_type", biz_type)
        if new_biz_type != biz_type:
            biz_type = new_biz_type
            logger.info(f"根据检索结果，动态切换 Prompt 模板至: [{biz_type}]")
            prompt_data = get_prompt_config(biz_type)

    llm = LLMFactory.get_model(
        prompt_data.get("config", {}),
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_data["content"]),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    async for chunk in (prompt_template | llm).astream({
        "input": user_input,
        "history": history,
        "context": context
    }):
        yield chunk

    if context.strip() and sources:
        source_text = "\n\n> **参考来源**：" + "，".join(f"`{s}`" for s in sources)
        yield AIMessageChunk(**{"content": source_text})

    yield AIMessageChunk(**{
        "content": "",
        "additional_kwargs": {
            "sources": sources,
            "biz_type": biz_type,
            "has_context": bool(context)
        }
    })
