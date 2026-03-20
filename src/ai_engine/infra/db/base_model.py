# src/ai_engine/infra/db/base_model.py
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field


class BaseModel(SQLModel):
    """
    所有表的基础模型
    """
    id: UUID = Field(default_factory=uuid4, primary_key=True)
