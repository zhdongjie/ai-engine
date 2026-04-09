# src/ai_engine/knowledge/explainers/special_chunk_explainer.py
import asyncio
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.factory import get_llm_model


class SpecialChunkExplainer:
    """
    语义增强解释器
    """

    def __init__(self):
        # 预加载不同业务的配置
        self.configs = {
            "code": get_prompt_config("code_explainer"),
            "table": get_prompt_config("table_explainer"),
            "math": get_prompt_config("math_explainer")
        }

    def _get_runner(self, chunk_type: str):
        """动态构建Runnable"""
        cfg = self.configs.get(chunk_type)
        if not cfg:
            return None

        llm = get_llm_model(cfg.get("config", {}))
        prompt = ChatPromptTemplate.from_template(cfg["content"] + "\n\n目标内容：\n{content}\n\n上下文信息：\n{context}")

        return prompt | llm | StrOutputParser()

    async def explain_doc(self, doc: Document, context: str) -> Document:
        """执行异步解释任务"""
        # 根据 Protector 注入的元数据判定类型
        chunk_type = None
        if doc.metadata.get("contains_code"):
            chunk_type = "code"
        elif doc.metadata.get("contains_table"):
            chunk_type = "table"
        elif doc.metadata.get("contains_math"):
            chunk_type = "math"

        if not chunk_type:
            return doc

        runner = self._get_runner(chunk_type)
        if runner:
            try:
                explanation = await runner.ainvoke({
                    "content": doc.page_content,
                    "context": context
                })
                # 将解释注入正文，提升检索召回
                doc.page_content = f"[AI Semantic Explanation]: {explanation}\n{doc.page_content}"
                doc.metadata["is_ai_enhanced"] = True
            except Exception as e:
                logger.error(f"AIOS Explainer 增强失败 (类型: {chunk_type}): {e}")

        return doc

    async def process_batch(self, docs: List[Document]):
        """并发处理，对齐你对 Ingestion Pipeline 性能的要求"""
        tasks = []
        for i, doc in enumerate(docs):
            # 获取滑动窗口上下文
            prev_c = docs[i - 1].page_content[:300] if i > 0 else ""
            next_c = docs[i + 1].page_content[:300] if i < len(docs) - 1 else ""
            window_context = f"PREV: {prev_c}\nNEXT: {next_c}"

            tasks.append(self.explain_doc(doc, window_context))

        return await asyncio.gather(*tasks)