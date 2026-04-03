# src/ai_engine/knowledge/document_loader.py
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from ai_engine.core.kb_manager import kb_manager
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.knowledge.loader_utils import build_text_splitter, build_markdown_splitter
from ai_engine.knowledge.loaders.markdown_loader import process_markdown
from ai_engine.knowledge.loaders.pdf_loader import process_pdf
from ai_engine.knowledge.loaders.txt_loader import process_txt
from ai_engine.knowledge.processors.factory import get_processor


def load_documents() -> List[Document]:
    markdown_splitter = build_markdown_splitter()
    text_splitter = build_text_splitter()

    all_docs: List[Document] = []
    knowledge_root = Path(settings.knowledge_dir)

    if not knowledge_root.exists():
        logger.error(f"Knowledge directory does not exist: {knowledge_root}")
        return []

    mode = settings.KB_INIT_MODE.lower()

    for biz_type, kb_config in kb_manager.registry.items():
        logger.info(f"Parsing business knowledge for KB: [{biz_type}]")
        processor = get_processor(biz_type)

        knowledge_config = kb_config.get("knowledge_path", biz_type)
        if isinstance(knowledge_config, dict):
            lang_path_map = knowledge_config
        else:
            lang_path_map = {"zh": knowledge_config}  # 默认兜底

        for lang, path_suffix in lang_path_map.items():
            biz_dir = knowledge_root / path_suffix

            if not biz_dir.exists() or not biz_dir.is_dir():
                logger.warning(f"KB [{biz_type}] lang [{lang}] Directory does not exist, skipping: {biz_dir}")
                continue

            logger.info(f"Loading docs for KB [{biz_type}] lang [{lang}] from {biz_dir}")

            # 1. 解析 Markdown 文件
            all_docs.extend(
                process_markdown(biz_dir, biz_type, lang, processor, text_splitter, markdown_splitter, mode)
            )

            # 2. 解析 TXT 文件
            all_docs.extend(
                process_txt(biz_dir, biz_type, lang, processor, text_splitter, mode)
            )

            # 3. 解析 PDF 文件
            all_docs.extend(
                process_pdf(biz_dir, biz_type, lang, processor, text_splitter, mode)
            )


    return all_docs
