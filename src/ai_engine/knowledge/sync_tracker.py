# src/ai_engine/knowledge/sync_tracker.py
import asyncio
import hashlib
from pathlib import Path
from typing import Dict

from sqlmodel import select

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.models.knowledge_sync import KnowledgeDocumentSync


class KBSyncTracker:
    def __init__(self):
        # 记录本次运行扫描到的所有待更新/新发现的文档信息
        self.pending_updates: Dict[str, dict] = {}

    @staticmethod
    def _calculate_md5(file_path: Path) -> str:
        """计算文件内容的 MD5"""
        hasher = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_path_md5(file_path: Path) -> str:
        """计算文件绝对路径的 MD5，作为跨平台的唯一标识"""
        path_str = str(file_path.resolve())
        return hashlib.md5(path_str.encode('utf-8')).hexdigest()

    async def inspect_document(self, file_path: Path, biz_type: str) -> str:
        """
        检查文档状态，决定是加载还是跳过，并维护待更新队列。
        返回: 'insert' (新增/覆盖), 'replace' (修改), 'skip' (跳过)
        """
        path_md5 = self.get_path_md5(file_path)
        current_hash = await asyncio.to_thread(self._calculate_md5, file_path)
        mode = settings.KB_INIT_MODE.lower()

        # 准备元数据信息
        file_info = {
            "file_path": str(file_path.resolve()),
            "biz_type": biz_type,
            "content_hash": current_hash
        }

        # 场景 A: 如果是全量覆盖模式，直接记录并返回 insert，不查数据库对比
        if mode == "overwrite":
            self.pending_updates[path_md5] = file_info
            return "insert"

        # 场景 B: 增量模式，需要与数据库进行 Source of Truth 对比
        async with db_manager.async_session_context() as session:
            statement = select(KnowledgeDocumentSync).where(
                KnowledgeDocumentSync.path_md5 == path_md5,
                KnowledgeDocumentSync.is_deleted == False
            )
            result = await session.execute(statement)
            record = result.scalar_one_or_none()

            # 1. 数据库没记录 -> 新文档
            if not record:
                logger.info(f"发现新文档: [{biz_type}] {file_path.name}")
                self.pending_updates[path_md5] = file_info
                return "insert"

            # 2. 数据库有记录但 MD5 不一致 -> 已修改
            if record.content_hash != current_hash:
                logger.info(f"文档已修改: [{biz_type}] {file_path.name}")
                self.pending_updates[path_md5] = file_info
                return "replace"

            # 3. 完全一致 -> 跳过
            return "skip"

    async def mark_sync_completed(self, docs: list):
        """
        当向量库 add_documents 成功后调用。
        将 pending_updates 队列中的状态正式持久化到 PostgreSQL。
        """
        if not self.pending_updates:
            logger.info("没有需要持久化的同步状态。")
            return

        # 1. 统计本次入库的文档切片数量
        chunk_counts = {}
        for doc in docs:
            p_md5 = doc.metadata.get("path_md5")
            if p_md5:
                chunk_counts[p_md5] = chunk_counts.get(p_md5, 0) + 1

        # 2. 批量写入数据库
        async with db_manager.async_session_context() as session:
            for path_md5, data in self.pending_updates.items():
                statement = select(KnowledgeDocumentSync).where(
                    KnowledgeDocumentSync.path_md5 == path_md5,
                    KnowledgeDocumentSync.is_deleted == False
                )
                result = await session.execute(statement)
                record = result.scalar_one_or_none()

                if not record:
                    record = KnowledgeDocumentSync(
                        path_md5=path_md5,
                        file_path=data["file_path"],
                        biz_type=data["biz_type"],
                        content_hash=data["content_hash"],
                        chunk_count=chunk_counts.get(path_md5, 0)
                    )
                    session.add(record)
                else:
                    # 更新现有记录
                    record.file_path = data["file_path"]
                    record.biz_type = data["biz_type"]
                    record.content_hash = data["content_hash"]
                    record.chunk_count = chunk_counts.get(path_md5, 0)
                    session.add(record)

            await session.commit()
            logger.success(f"成功将 {len(self.pending_updates)} 个文档的指纹同步至 PostgreSQL")

        # 3. 清空队列，防止污染下次运行
        self.pending_updates.clear()


# 单例导出
sync_tracker = KBSyncTracker()
