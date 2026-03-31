# src/ai_engine/rag_plugins/java_docs_plugin.py
from typing import List, Dict, Any

from langchain_core.runnables import RunnableConfig

from ai_engine.core.logger import logger
from .base import BaseRAGPlugin


def _attach_questions(context: str, questions: List[Dict[str, Any]], lang: str) -> str:
    if not questions:
        return context

    i18n_config = {
        "zh": {
            "header": ["\n---", "🎯 **实战演练**", "学完理论后，建议完成以下练习："],
            "link_text": "开始练习"
        },
        "en": {
            "header": ["\n---", "🎯 **Practice Exercises**",
                       "After learning the theory, we recommend completing these exercises:"],
            "link_text": "Start Practice"
        },
        "cht": {
            "header": ["\n---", "🎯 **實戰演練**", "學完理論後，建議完成以下練習："],
            "link_text": "開始練習"
        },
    }

    config = i18n_config.get(lang, i18n_config["zh"])

    lines = list(config["header"])
    link_label = config["link_text"]

    for i, q in enumerate(questions, 1):
        title = q.get("title", "Untitled")
        url = q.get("url", "")

        if not url:
            continue

        lines.append(f"{i}. {title}")
        lines.append(f"   👉 [{link_label}]({url})")

    return f"{context.strip()}\n\n" + "\n".join(lines)


class JavaDocsPlugin(BaseRAGPlugin):
    """Java 文档课后练习题解析与注入插件"""

    def process(
            self,
            docs: List[Any],
            context: str,
            extra: Dict[str, Any],
            config: RunnableConfig
    ):
        configurable = config.get("configurable") or {}
        user_lang = configurable.get("lang", "zh")
        questions = []

        import json
        from json import JSONDecodeError

        for doc in docs:
            q_raw = doc.metadata.get("questions", "[]")

            try:
                if isinstance(q_raw, str):
                    q_list = json.loads(q_raw)
                else:
                    q_list = q_raw
                if isinstance(q_list, list):
                    questions.extend(q_list)
                else:
                    logger.warning(f"文档元数据 questions 格式异常，期望 list 但得到 {type(q_list)}")

            except (JSONDecodeError, TypeError) as e:
                doc_id = doc.metadata.get("id", "unknown")
                logger.error(f"解析文档 [{doc_id}] 的推荐问题失败: {e}")
                continue

        # 去重
        seen = set()
        unique = []

        for q in questions:
            url = q.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(q)

        unique = unique[:3]  # 最多推 3 道题

        extra["questions"] = unique

        # Markdown 拼接 context
        context = _attach_questions(context, unique, user_lang)

        return context, extra
