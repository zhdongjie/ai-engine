# src/ai_engine/rag_plugins/java_docs_plugin.py
import json

from .base import BaseRAGPlugin


def _attach_questions(context: str, questions: list[dict]) -> str:
    if not questions:
        return context

    lines = [
        "\n\n---",
        "🎯 **实战演练**",
        "学完理论后，建议完成以下练习："
    ]

    for i, q in enumerate(questions, 1):
        title = q.get("title", "")
        url = q.get("url", "")

        if not url:
            continue

        lines.append(f"{i}. {title}")
        lines.append(f"👉 [开始练习]({url})")

    return context + "\n" + "\n".join(lines)


class JavaDocsPlugin(BaseRAGPlugin):
    """Java 文档课后练习题解析与注入插件"""

    def process(self, docs, context, extra):
        questions = []

        for doc in docs:
            # 兼容处理：防范部分脏数据
            q_raw = doc.metadata.get("questions", "[]")
            try:
                # 反序列化 JSON 字符串为 Python List
                q_list = json.loads(q_raw) if isinstance(q_raw, str) else q_raw
                if isinstance(q_list, list):
                    questions.extend(q_list)
            except Exception:
                continue

        # 去重逻辑 (根据 URL 去重)
        seen = set()
        unique = []

        for q in questions:
            url = q.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(q)

        # 限制最多推 3 道题
        unique = unique[:3]

        # 将结构化数据塞入 extra，最终会通过 additional_kwargs 传给前端
        extra["questions"] = unique

        # 将 Markdown 格式的题目拼接到传给大模型的 context 尾部
        context = _attach_questions(context, unique)

        return context, extra
