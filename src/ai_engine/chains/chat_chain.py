# src/ai_engine/chains/chat_chain.py
import uuid
from typing import List, Tuple

from dashscope import TextReRank
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
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
    # 🔥 建议前端也把 session_id 传在 query 或者 header 里，LangServe 默认使用 header 或 config 取
    # session_id: str = Field(..., description="会话ID") 


# --- 1. 全局 Embedding 初始化 ---
embeddings = OpenAIEmbeddings(
    api_key=settings.QWEN_API_KEY,
    base_url=settings.QWEN_API_BASE,
    model=settings.QWEN_MODEL_EMBEDDING,
    check_embedding_ctx_length=False
)


# --- 2. 增强型工具函数 (保持不变) ---
def get_reranked_docs(query: str, initial_docs: list) -> list:
    # ... (你的原代码保持完全不变) ...
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


def format_docs_with_sources(docs) -> Tuple[str, List[str]]:
    # ... (你的原代码保持完全不变) ...
    if not docs: return "", []
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted(list(set(doc.metadata.get("file_name", "未知文档") for doc in docs)))
    return context, sources


# --- 3. 核心逻辑：⚡ 纯异步化改造 ---
# 注意这里变成了 async def
async def adynamic_rag_run(input_data: dict) -> AIMessage:
    """原子化执行：海选 -> 精选(Rerank) -> 组装 -> 运行"""
    biz_type = input_data.get("biz_type", "virtual_card")
    user_input = input_data.get("input")
    history = input_data.get("history", [])

    # A. 异步海选 (使用 ainvoke)
    vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings
    )
    # 强制改为异步检索，防止阻塞
    initial_docs = await vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": {"biz_type": biz_type}}
    ).ainvoke(user_input)

    # B. 精选 (Dashscope 目前是同步 API，这里直接调，如果追求极致性能未来可以用 asyncio.to_thread 包装)
    final_docs = get_reranked_docs(user_input, initial_docs)

    # C. 格式化
    context, sources = format_docs_with_sources(final_docs)

    # D. 获取业务配置与 Prompt
    prompt_data = get_prompt_config(biz_type)
    instruction = prompt_data["content"]
    biz_config = prompt_data["config"]

    # E. 实例化局部 LLM
    llm = ChatOpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_API_BASE,
        model=biz_config.get("model", settings.QWEN_MODEL_LLM),
        temperature=biz_config.get("temperature", settings.TEMPERATURE),
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # F. 异步执行 LLM (使用 ainvoke)
    chain = prompt_template | llm
    response_msg = await chain.ainvoke({
        "input": user_input,
        "history": history,
        "context": context
    })
    # 1. 提取文本内容
    answer = response_msg.content

    # 2. 提取 Token 消耗数据
    token_usage = response_msg.usage_metadata or {}
    if not token_usage and "token_usage" in response_msg.response_metadata:
        token_usage = response_msg.response_metadata["token_usage"]

    # G. 智能追加来源
    if context.strip() and sources:
        answer += "\n\n> 💡 **参考来源**：" + "，".join(f"`{s}`" for s in sources)

    return AIMessage(
        content=answer,
        additional_kwargs={
            "sources": sources,  # 把文档来源数组存进元数据
            "biz_type": biz_type,  # 把业务路由标识存进元数据
            "has_context": bool(context),  # 标记这次回答是否真的用到了知识库
            "token_usage": token_usage  # 记录Token消耗
        }
    )


# --- 4. 组装 RAG 主链 ---
# LangChain 会自动识别这是个异步函数
rag_chain = RunnableLambda(adynamic_rag_run)


# --- 5. ⚡ 真实的 PostgreSQL 永久记忆接入 ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    当 LangServe 接收到请求时，会带着前端传来的 session_id 调用这个函数。
    我们直接返回连接到数据库的适配器。
    """
    # 简单的 UUID 格式校验与容错（如果前端没传标准的UUID，自动生成一个）
    try:
        uuid.UUID(session_id)
        valid_session_id = session_id
    except ValueError:
        logger.warning(f"接收到非法的 session_id: {session_id}，已自动替换为新 UUID")
        valid_session_id = str(uuid.uuid4())

    return PostgresAsyncChatMessageHistory(
        session_id=valid_session_id,
        tenant_id="default_tenant",  # 未来可从 request headers 提取
        user_id="default_user"  # 未来可从 request headers 提取
    )


# 最终导出的 Chain 对象
chat_chain = RunnableWithMessageHistory(
    rag_chain,  # type: ignore
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
).with_types(input_type=ChatInput)
