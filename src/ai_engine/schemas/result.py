# src/ai_engine/schemas/result.py
from typing import TypeVar, Generic, Optional

from pydantic import Field

from ai_engine.core.constants import ResponseCode
from ai_engine.schemas.base import BaseSchema

T = TypeVar("T")


class Result(BaseSchema, Generic[T]):
    """
    对齐 Java 规范的通用响应包装类
    """
    code: int = Field(ResponseCode.SUCCESS.value, description="业务状态码")
    msg: str = Field("success", description="提示消息")
    data: Optional[T] = Field(None, description="业务数据")

    @classmethod
    def success(cls, data: T = None, msg: str = "success"):
        return cls(code=ResponseCode.SUCCESS.value, msg=msg, data=data)

    @classmethod
    def fail(cls, code: int, msg: str):
        return cls(code=code, msg=msg, data=None)
