# src/ai_engine/infra/db/mixins.py
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime
from sqlmodel import Field


def utc_now():
    """获取当前标准的 UTC 时间（带时区信息），防止服务器物理机时钟不一致导致的时间混乱"""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """
    时间戳 Mixin：
    为模型自动注入创建时间和更新时间。适用于几乎所有需要生命周期追踪的业务表。
    """
    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),  # type: ignore  # 强制底层 PostgreSQL 使用带时区的 TIMESTAMP WITH TIME ZONE
        nullable=False
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),  # type: ignore
        nullable=False
    )


class SoftDeleteMixin:
    """
    软删除 Mixin：
    提供逻辑删除能力，代替物理删除 (DELETE FROM)。
    在企业级应用中，数据通常只做状态变更，以保留审计凭据和历史上下文。
    """
    is_deleted: bool = Field(
        default=False,
        index=True  # 建立索引，因为绝大部分的 SELECT 查询都需要带上 is_deleted=False 条件
    )


class TenantMixin:
    """
    多租户 Mixin (SaaS 架构核心)：
    用于实现数据层面的租户隔离 (Tenant Isolation)。
    在微服务或平台级应用中，通过此字段确保不同企业/组织的数据在物理表中共存时不会发生越权访问。
    """
    tenant_id: str = Field(
        index=True,  # 建立索引，因为基于租户的数据过滤是最高频的操作
        default_factory=lambda: str(uuid.uuid4())
    )


class AuditMixin:
    """
    操作审计 Mixin：
    记录数据的操作主体，追踪 "是谁创建的" 以及 "是谁最后修改的"。
    满足企业级权限管控、操作留痕以及合规审查 (Compliance) 的需求。
    """
    created_by: Optional[str] = Field(
        default=None,
        index=True  # 方便后台管理员通过 user_id 倒查某个人创建的所有资源
    )

    updated_by: Optional[str] = Field(
        default=None,
        index=True
    )
