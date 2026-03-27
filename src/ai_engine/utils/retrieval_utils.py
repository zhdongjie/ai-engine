# src/ai_engine/utils/retrieval_utils.py
from typing import List, Tuple

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.llm.llm_factory import LLMFactory


def get_reranked_docs(query: str, initial_docs: list) -> list:
    """针对 gte-rerank-v2 优化的重排函数"""
    if not initial_docs:
        return []

    documents_text = [doc.page_content for doc in initial_docs]
    try:
        resp = LLMFactory.call_rerank(
            query=query,
            documents=documents_text,
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
