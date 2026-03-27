# src/ai_engine/schemas/base.py
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """
    底层基类：负责处理 snake_case 到 camelCase 的转换
    """
    model_config = ConfigDict(
        alias_generator=to_camel,  # 给外部（Java/前端）看驼峰
        populate_by_name=True,  # 内部支持按名称填充
        from_attributes=True  # 支持从 ORM 对象转换
    )
