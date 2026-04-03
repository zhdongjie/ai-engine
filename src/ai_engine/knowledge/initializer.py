# src/ai_engine/knowledge/initializer.py
from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.embedding.factory import get_embedding_provider
from ai_engine.infra.vector_store.factory import VectorStoreFactory
from ai_engine.knowledge.document_loader import load_documents
from ai_engine.knowledge.sync_tracker import sync_tracker


def run_init():
    mode = settings.KB_INIT_MODE.lower()
    if mode == "skip":
        return

    logger.info(f"知识库初始化启动, 模式: [{mode.upper()}]")

    # 1. 准备 Embedding
    embeddings = get_embedding_provider()

    # 2. 获取 Provider
    provider = VectorStoreFactory.get_provider(embeddings)

    # 3. 处理全量覆盖
    if mode == "overwrite":
        provider.clear_all()

    # 4. 加载文档
    docs = load_documents()
    if not docs:
        logger.info("无文件需更新。")
        return

    try:
        # 5. 增量清理
        if mode == "incremental":
            for p_md5 in sync_tracker.pending_updates.keys():
                provider.delete_by_path_md5(p_md5)

        # 6. 写入数据
        logger.info(f"正在写入 {len(docs)} 个切片...")
        provider.add_documents(docs)

        # 7. 提交状态
        sync_tracker.mark_sync_completed(docs)
        logger.success("知识库初始化成功！")

    except Exception as e:
        logger.error(f"同步失败: {e}")
        raise
