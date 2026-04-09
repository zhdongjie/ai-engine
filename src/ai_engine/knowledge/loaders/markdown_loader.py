# src/ai_engine/knowledge/loaders/markdown_loader.py
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ai_engine.core.logger import logger
from ai_engine.knowledge.loader_utils import (
    split_documents_for_ingestion,
    enrich_chunk_metadata,
    apply_document_augmentation
)
from ai_engine.knowledge.sync_tracker import sync_tracker


class MarkdownProtector:
    """Markdown 结构保护器：支持代码、表格、链接、图片、公式、图表"""

    def __init__(self):
        self.mapping: Dict[str, Dict[str, Any]] = {}

        # 1. 块级结构正则
        self.BLOCK_PATTERNS = {
            # 代码块与图表
            "code_or_diagram": re.compile(r"```[ \t]*(\w*)\n([\s\S]*?)```"),
            # 块级公式 $$...$$
            "math_block": re.compile(r"\$\$\s*([\s\S]+?)\s*\$\$"),
            # 表格
            "table": re.compile(
                r"(?:^|\n)\|?([^\n]+\|)+[^\n]*\n\|?([ \t]*:?-+:?[ \t]*\|)+[^\n]*(\n\|?([^\n]+\|)+[^\n]*)*"),
        }

        self.INLINE_PATTERNS = {
            # 行内公式 $...$
            "math_inline": re.compile(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$"),
            # 图片 ![alt](url)
            "image": re.compile(r"!\[([^]]*?)]\(([^)]+?)\)"),
            # 链接 [text](url)
            "link": re.compile(r"(?<!!)\[([^]]*?)]\(([^)]+?)\)"),
        }

        # 还原扫描正则
        self.RESTORE_PATTERN = re.compile(
            r"__PROTECTED_(CODE_OR_DIAGRAM|MATH_BLOCK|TABLE|MATH_INLINE|IMAGE|LINK)_[0-9a-f]{32}(?:_\w+)?__")

    def protect(self, text: str) -> str:
        # 先处理大块，再处理行内，防止冲突
        for cat, pattern in self.BLOCK_PATTERNS.items():
            text = pattern.sub(lambda m: self._map_match(cat, m), text)
        for cat, pattern in self.INLINE_PATTERNS.items():
            text = pattern.sub(lambda m: self._map_match(cat, m), text)
        return text

    def _map_match(self, category: str, match: re.Match) -> str:
        uid_base = f"__PROTECTED_{category.upper()}_{uuid.uuid4().hex}"
        content = match.group(0)
        data = {"type": category, "content": content}

        if category == "code_or_diagram":
            lang = (match.group(1) or "text").lower()
            data["lang"] = lang
            # 识别是否是 Mermaid 等图表
            if lang in ["mermaid", "flowchart", "plantuml"]:
                data["type"] = "diagram"
            uid_base += f"_{lang}"

        uid = f"{uid_base}__"
        self.mapping[uid] = data

        # 块状结构前后加换行，保护语义边界
        return f"\n\n{uid}\n\n" if category in self.BLOCK_PATTERNS else uid

    def restore_and_enrich(self, doc: Document) -> Document:
        stats = {"has_table": False, "has_code": False, "has_math": False, "has_diagram": False, "langs": set()}

        def _callback(m):
            uid = m.group(0)
            if uid not in self.mapping: return uid
            d = self.mapping[uid]
            # 统计信息，用于下面的元数据注入
            if d["type"] == "table":
                stats["has_table"] = True
            elif d["type"] == "code_or_diagram":
                stats["has_code"] = True
                stats["langs"].add(d.get("lang"))
            elif d["type"] == "diagram":
                stats["has_diagram"] = True
            elif "math" in d["type"]:
                stats["has_math"] = True
            return d["content"]

        doc.page_content = self.RESTORE_PATTERN.sub(_callback, doc.page_content)

        # 注入元数据
        if stats["has_table"]: doc.metadata["contains_table"] = True
        if stats["has_math"]: doc.metadata["contains_math"] = True
        if stats["has_code"]:
            doc.metadata["contains_code"] = True
            doc.metadata["code_languages"] = list(stats["langs"])
        if stats["has_diagram"]: doc.metadata["contains_diagram"] = True

        return doc


def process_markdown(
        biz_dir: Path, biz_type: str, lang: str, processor,
        text_splitter: RecursiveCharacterTextSplitter,
        markdown_splitter: MarkdownHeaderTextSplitter, mode: str
) -> List[Document]:
    docs = []
    for markdown_path in biz_dir.glob("**/*.md"):
        try:
            action = sync_tracker.inspect_document(markdown_path, biz_type)
            if mode == "incremental" and action == "skip":
                continue

            content = markdown_path.read_text(encoding="utf-8")
            path_md5 = sync_tracker.get_path_md5(markdown_path)
            extracted_meta = {"lang": lang, "path_md5": path_md5}

            if processor:
                content, proc_meta = processor.process(content, markdown_path)
                extracted_meta.update(proc_meta)

            # 1. 预保护 ---
            protector = MarkdownProtector()
            protected_content = protector.protect(content)

            # 2. 基于 Header 切分 ---
            markdown_splits = markdown_splitter.split_text(protected_content)

            # 3. 语义/Token 二次切分 ---
            final_splits, chunk_strategy = split_documents_for_ingestion(markdown_splits, text_splitter)

            # 4. 还原与注入元数据 ---
            for doc in final_splits:
                protector.restore_and_enrich(doc)

            # 5. 补充通用元数据与增强 ---
            enrich_chunk_metadata(
                final_splits, biz_type=biz_type, file_name=markdown_path.name,
                source_type="markdown", chunk_strategy=chunk_strategy,
            )

            for doc in final_splits:
                doc.metadata.update(extracted_meta)

            apply_document_augmentation(final_splits)
            docs.extend(final_splits)

        except Exception as exc:
            logger.error(f"Failed to parse markdown file {markdown_path}: {exc}")

    return docs
