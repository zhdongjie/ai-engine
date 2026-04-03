# src/ai_engine/models/knowledge_sync.py
from sqlmodel import Field

from ai_engine.infra.db.base_model import BaseModel
from ai_engine.infra.db.mixins import (
    TimestampMixin,
    TenantMixin,
    SoftDeleteMixin
)


class KnowledgeDocumentSync(
    BaseModel,
    TimestampMixin,
    SoftDeleteMixin,
    TenantMixin,
    table=True
):
    __tablename__ = "knowledge_document_sync"

    path_md5: str = Field(
        max_length=64,
        index=True,
        unique=True,
        description="文件路径的 MD5 (用于唯一标识物理文件)"
    )

    file_path: str = Field(
        max_length=1024,
        description="文件绝对路径"
    )

    biz_type: str = Field(
        max_length=128,
        index=True,
        description="业务归属 (如 java_tutor)"
    )

    content_hash: str = Field(
        max_length=64,
        description="文件内容的 MD5"
    )

    chunk_count: int = Field(
        default=0,
        description="切分后的 Chunk 数量"
    )
