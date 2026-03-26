# src/ai_engine/chains/chat_chain.py
import asyncio
import uuid
from typing import List, Tuple, AsyncIterator, Dict, Any

from dashscope import TextReRank
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, ConfigurableFieldSpec, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.infra.llm.message_adapter import PostgresAsyncChatMessageHistory


# --- 0. 输入模型定义 ---
class ChatInput(BaseModel):
    input: str = Field(..., description="用户的纯文本提问")
    biz_type: str = Field(default="normal_chat", description="业务类型标识符")


# --- 1. 全局 Embedding 初始化 ---
embeddings = OpenAIEmbeddings(
    api_key=settings.QWEN_API_KEY.get_secret_value(),
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
            api_key=settings.QWEN_API_KEY.get_secret_value(),
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


# --- 3. 核心双轨逻辑 A：纯净闲聊生成 (不查向量库) ---
async def anormal_chat_run(input_data: Dict[str, Any]) -> AsyncIterator[BaseMessage]:
    """处理打招呼、写代码、常识等无需查库的纯通用对话"""
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])
    logger.debug("进入 Normal Chat 模式，直接利用大模型本体能力回答...")

    # 1. 动态获取闲聊专属的 Prompt 和配置 (彻底消灭硬编码！)
    prompt_data = get_prompt_config("normal_chat")

    # 2. 实例化局部 LLM (优先读取 YAML 里调高的温度，让闲聊更有创意)
    llm = ChatOpenAI(
        api_key=settings.QWEN_API_KEY.get_secret_value(),
        base_url=settings.QWEN_API_BASE,
        model=prompt_data["config"].get("model", settings.QWEN_MODEL_LLM),
        temperature=prompt_data["config"].get("temperature", 0.7),
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    # 3. 使用 YAML 中的 content 组装系统提示词
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_data["content"]),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    async for chunk in (prompt_template | llm).astream({
        "input": user_input,
        "history": history
    }):
        yield chunk

    # 注入元数据 (供 message_adapter 记录)
    meta_payload = {
        "content": "",
        "additional_kwargs": {
            "sources": [],
            "biz_type": "normal_chat",
            "has_context": False
        }
    }
    yield AIMessageChunk(**meta_payload)


# --- 4. 核心双轨逻辑 B：RAG 业务增强生成 (查库) ---
async def adynamic_rag_run(input_data: Dict[str, Any]) -> AsyncIterator[BaseMessage]:
    """
    原子化执行：海选 -> 精选(Rerank) -> 组装 -> 运行 (支持流式产出)
    """
    biz_type = input_data.get("biz_type", "normal_chat")
    user_input = input_data.get("input", "")
    history = input_data.get("history", [])
    logger.debug(f"进入 RAG 模式，开始检索知识库 ({biz_type})...")

    # A. 异步海选
    if settings.VECTOR_STORE_TYPE.lower() == "postgresql":
        from langchain_postgres import PGVector
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="ai_knowledge_base",
            connection=db_manager.engine,
            use_jsonb=True,
            create_extension=False,
        )
        logger.debug(f"检索底层引擎: PostgreSQL | biz_type: {biz_type}")
    else:
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings
        )
        logger.debug(f"检索底层引擎: ChromaDB")

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": settings.VECTOR_SEARCH_TOP_K
        }
    )

    initial_docs = await retriever.ainvoke(user_input)
    logger.debug(f"召回阶段完成，原始文档数: {len(initial_docs)}")

    # B. 精选
    final_docs = await asyncio.to_thread(get_reranked_docs, user_input, initial_docs)
    logger.info(f"重排阶段完成，剩余精选文档: {len(final_docs)}")

    # C. 格式化
    context, sources = format_docs_with_sources(final_docs)

    if final_docs:
        biz_type = final_docs[0].metadata.get("biz_type", biz_type)
        logger.info(f"💡 根据检索结果，动态切换 Prompt 模板至: [{biz_type}]")

    # D. 获取业务配置与 Prompt
    prompt_data = get_prompt_config(biz_type)

    # E. 实例化局部 LLM
    llm = ChatOpenAI(
        api_key=settings.QWEN_API_KEY.get_secret_value(),
        base_url=settings.QWEN_API_BASE,
        model=prompt_data["config"].get("model", settings.QWEN_MODEL_LLM),
        temperature=prompt_data["config"].get("temperature", settings.TEMPERATURE),
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
        source_text = "\n\n> **参考来源**：" + "，".join(f"`{s}`" for s in sources)
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


# ==========================================
# 5. 意图分类器 (Router) & 核心调度中枢
# ==========================================
# A. 初始化轻量级分类大模型
router_config = get_prompt_config("intent_router")
router_llm = ChatOpenAI(
    model=router_config.get("config", {}).get("model", "qwen-turbo"),
    api_key=settings.QWEN_API_KEY.get_secret_value(),
    base_url=settings.QWEN_API_BASE,
    temperature=router_config.get("config", {}).get("temperature", 0.0),
    max_tokens=router_config.get("config", {}).get("max_tokens", 10)
)
router_prompt = PromptTemplate.from_template(router_config["content"])
intent_router = router_prompt | router_llm | StrOutputParser()


def route_logger(info: dict) -> dict:
    """日志探针：观察路由器的决策结果"""
    intent = info.get("intent", "UNKNOWN").strip().upper()
    logger.info(f"路由判定结果: [{intent}] | 用户输入: {info.get('input', '')[:15]}...")
    return info


# ------------------------------------------
# 强类型动态路由函数
# ------------------------------------------
def route_logic(info: dict):
    """
    根据 intent 动态返回下一阶段的 Runnable 链路。
    LangChain 会自动将当前上下文无缝传递给被返回的链路。
    """
    if "RAG" in info.get("intent", "").upper():
        return RunnableLambda(adynamic_rag_run)

    # 默认兜底分支
    return RunnableLambda(anormal_chat_run)


# B. 构建带交通管制功能的主链 (彻底抛弃 RunnableBranch)
master_chain = (
    # 1. 提取用户的输入进行意图分析，同时保留原始入参
        RunnablePassthrough.assign(intent=intent_router)
        # 2. 打印决策日志
        | RunnableLambda(route_logger)
        # 3. 核心分流：直接让 RunnableLambda 根据逻辑返回具体的子链
        | RunnableLambda(route_logic)
)

# --- 6. 真实的 PostgreSQL 永久记忆接入 ---
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
    master_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="session_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="tenant_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="user_id", annotation=str, is_shared=True),
    ]
).with_types(input_type=ChatInput)
