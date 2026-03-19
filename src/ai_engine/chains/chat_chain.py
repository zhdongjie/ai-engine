from typing import List, Tuple

from dashscope import TextReRank  # 引入阿里重排 SDK
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.core.settings import settings


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
    """
    针对 gte-rerank-v2 优化的重排函数
    """
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
            score = getattr(item, 'relevance_score', None)
            index = getattr(item, 'index', None)

            # 如果是字典格式（保险起见）
            if score is None and isinstance(item, dict):
                score = item.get('relevance_score')
                index = item.get('index')

            if index is not None and score is not None:
                # 使用 settings 中配置的阈值
                if score > settings.RERANK_THRESHOLD:
                    original_doc = initial_docs[index]
                    original_doc.metadata["rerank_score"] = score
                    final_docs.append(original_doc)
            else:
                logger.warning(f"跳过无法解析的重排条目: {item}")

        return final_docs

    except Exception as e:
        logger.error(f"Rerank 过程异常: {e}")
        return initial_docs[:2]


def format_docs_with_sources(docs) -> Tuple[str, List[str]]:
    """同时格式化文档内容和提取不重复的文件来源"""
    if not docs:
        return "", []

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted(list(set(
        doc.metadata.get("file_name", "未知文档") for doc in docs
    )))
    return context, sources


# --- 3. 核心逻辑：动态运行函数 ---

def dynamic_rag_run(input_data: dict):
    """原子化执行：海选 -> 精选(Rerank) -> 组装 -> 运行"""
    biz_type = input_data.get("biz_type", "virtual_card")
    user_input = input_data.get("input")
    history = input_data.get("history", [])

    # A. 第一步：海选 (向量检索)
    # 调大 k 值（比如 10），给 Rerank 留下足够的筛选空间
    vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings
    )
    initial_docs = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {"biz_type": biz_type}
        }
    ).invoke(user_input)

    # B. 第二步：精选 (gte-rerank-v2)
    # 让考官从 10 个里挑出真正能回答问题的文档
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
        streaming=True
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # F. 执行 LLM
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({
        "input": user_input,
        "history": history,
        "context": context
    })

    # G. 智能追加来源
    if context.strip() and sources:
        answer += "\n\n> 💡 **参考来源**：" + "，".join(f"`{s}`" for s in sources)

    return answer


# --- 4. 组装 RAG 主链 ---
rag_chain = RunnableLambda(dynamic_rag_run)

# --- 5. 记忆组件与会话管理 ---
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 最终导出的 Chain 对象
chat_chain = RunnableWithMessageHistory(
    rag_chain,  # type: ignore
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
).with_types(input_type=ChatInput)
