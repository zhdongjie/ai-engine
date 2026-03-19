import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ai_engine.core.settings import settings
from ai_engine.core.logger import logger


def run_init():
    logger.info("🚀 [Task] 开始知识库初始化...")

    # 1. 初始化组件
    embeddings = OpenAIEmbeddings(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_API_BASE,
        model=settings.QWEN_MODEL_EMBEDDING,
        check_embedding_ctx_length=False,
        chunk_size=10
    )

    # 定义 MD 切分规则：按一级、二级、三级标题切
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 二次切分：防止某个章节内容过长，超过 Embedding 模型限制
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 金融文档通常较严谨，可以稍微大一点
        chunk_overlap=100,  # 增加重叠，减少语义丢失
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]  # 优先按句号切，而不是字数
    )
    # 2. 清理旧数据
    persist_dir = settings.chroma_persist_dir
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    all_docs = []
    kb_root = Path(settings.knowledge_dir)

    # 3. 遍历业务文件夹
    for biz_dir in kb_root.iterdir():
        if not biz_dir.is_dir(): continue
        biz_type = biz_dir.name
        logger.info(f"📂 正在深度解析业务模块: [ {biz_type} ]")

        # --- 分类处理文档 ---

        # A. 处理 MD 文件 (智能切片)
        md_files = list(biz_dir.glob("**/*.md"))
        for md_path in md_files:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 第一步：按标题切分
            md_header_splits = md_splitter.split_text(content)
            # 第二步：对长章节进行二次微切
            md_final_splits = text_splitter.split_documents(md_header_splits)

            for doc in md_final_splits:
                doc.metadata["biz_type"] = biz_type
                doc.metadata["file_name"] = md_path.name
                all_docs.append(doc)

        # B. 处理 TXT 和 PDF (常规切片)
        txt_loader = DirectoryLoader(str(biz_dir), glob="**/*.txt", loader_cls=TextLoader,
                                     loader_kwargs={'encoding': 'utf-8'})
        pdf_loader = DirectoryLoader(str(biz_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)  # type: ignore

        other_raw_docs = txt_loader.load() + pdf_loader.load()
        other_splits = text_splitter.split_documents(other_raw_docs)

        for doc in other_splits:
            if doc.page_content.strip():
                doc.metadata["biz_type"] = biz_type
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "unknown"))
                all_docs.append(doc)

    # 4. 写入数据库
    if all_docs:
        logger.info(f"💾 正在将 {len(all_docs)} 个优化后的片段写入 ChromaDB...")
        Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=persist_dir)
        logger.success("✨ 智能知识库构建完成！")


if __name__ == "__main__":
    run_init()