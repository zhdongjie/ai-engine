# src/ai_engine/chains/chat_chain.py
import asyncio
import uuid
from typing import List, Tuple, AsyncIterator, Dict, Any

from dashscope import TextReRank
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings
from ai_engine.infra.llm.message_adapter import PostgresAsyncChatMessageHistory


# --- 0. 输入模型定义 ---
class ChatInput(BaseModel):
    input: str = Field(..., description="用户的纯文本提问")
    biz_type: str = Field(default="virtual_card", description="业务类型标识符")


# --- 1. 全局 Embedding 初始化 ---
embeddings = OpenAIEmbeddings(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_API_BASE,
    model=settings.QWEN_MODEL_EMBEDDING,
    check_embedding_ctx_length=False
)


# --- 2. 增强型工具函数 ---
def get_reranked_docs(query: str, initial_docs: list) -> list:
    """针对 gte-rerank-v2 优化的重排函数"""
    if not initial_docs:
        return []

    documents_text = [doc.page_content for doc in initial_docs]
    try:
        resp = TextReRank.call(
            model=settings.QWEN_MODEL_RERANK,
            query=query,
            documents=documents_text,
            top_n=settings.RERANK_TOP_N,
            api_key=settings.QWEN_API_KEY
        )
        if resp.status_code != 200:
            logger.error(f"Rerank API 报错: {resp.message}")
            return initial_docs[:2]

        final_docs = []
        for item in resp.output.results:
            score = getattr(item, 'relevance_score', item.get('relevance_score') if isinstance(item, dict) else None)
            index = getattr(item, 'index', item.get('index') if isinstance(item, dict) else None)

            if index is not None and score is not None and score > settings.RERANK_THRESHOLD:
                original_doc = initial_docs[index]
                original_doc.metadata["rerank_score"] = score
                final_docs.append(original_doc)
        return final_docs
    except Exception as e:
        logger.error(f"Rerank 过程异常: {e}")
        return initial_docs[:2]


def format_docs_with_sources(docs: list) -> Tuple[str, List[str]]:
    """同时格式化文档内容和提取不重复的文件来源"""
    if not docs:
        return "", []
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted(list(set(doc.metadata.get("file_name", "未知文档") for doc in docs)))
    return context, sources


# --- 3. 核心逻辑：纯异步流式改造 ---
async def adynamic_rag_run(input_data: Dict[str, Any]) -> AsyncIterator[BaseMessage]:
    """
    原子化执行：海选 -> 精选(Rerank) -> 组装 -> 运行 (支持流式产出)
    """
    biz_type = input_data.get("biz_type", "virtual_card")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])

    # A. 异步海选 (使用 ainvoke 保持异步非阻塞)
    vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings
    )
    # 强制改为异步检索，防止阻塞
    initial_docs = await vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": {"biz_type": biz_type}}
    ).ainvoke(user_input)
    logger.debug(f"召回阶段完成，原始文档数: {len(initial_docs)}")

    # B. 精选
    final_docs = await asyncio.to_thread(get_reranked_docs, user_input, initial_docs)
    logger.info(f"重排阶段完成，剩余精选文档: {len(final_docs)}")

    # C. 格式化
    context, sources = format_docs_with_sources(final_docs)

    # D. 获取业务配置与 Prompt
    prompt_data = get_prompt_config(biz_type)

    # E. 实例化局部 LLM (开启流式与计费统计)
    llm = ChatOpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_API_BASE,
        model=prompt_data["config"].get("model", settings.QWEN_MODEL_LLM),
        temperature=settings.TEMPERATURE,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_data["content"]),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # F. 核心流式转发
    async for chunk in (prompt_template | llm).astream({
        "input": user_input,
        "history": history,
        "context": context
    }):
        yield chunk

    # G. 智能追加来源
    if context.strip() and sources:
        source_text = "\n\n> 💡 **参考来源**：" + "，".join(f"`{s}`" for s in sources)
        yield AIMessageChunk(**{"content": source_text})

    # H. 注入元数据
    meta_payload = {
        "content": "",
        "additional_kwargs": {
            "sources": sources,
            "biz_type": biz_type,
            "has_context": bool(context)
        }
    }
    yield AIMessageChunk(**meta_payload)


# --- 4. 组装 RAG 主链 ---
rag_chain = RunnableLambda(adynamic_rag_run)  # type: ignore


# --- 5. 真实的 PostgreSQL 永久记忆接入 ---
def get_session_history(session_id: str, tenant_id: str, user_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取或创建异步数据库记忆适配器"""
    try:
        uuid.UUID(session_id)
        valid_session_id = session_id
    except ValueError:
        logger.warning(f"接收到非法的 session_id: {session_id}，已自动替换为新 UUID")
        valid_session_id = str(uuid.uuid4())

    return PostgresAsyncChatMessageHistory(
        session_id=valid_session_id,
        tenant_id=tenant_id,
        user_id=user_id
    )


# 最终导出的具有持久化记忆的对话链对象
chat_chain = RunnableWithMessageHistory(
    rag_chain,  # type: ignore
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="session_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="tenant_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="user_id", annotation=str, is_shared=True),
    ]
).with_types(input_type=ChatInput)
