# scripts/processors/factory.py
from typing import Optional

from .base import BaseProcessor
from .java_docs_processor import JavaDocProcessor

_PROCESSOR_REGISTRY = {
    "java_tutor": JavaDocProcessor(enable_lang_detect=True),
}


def get_processor(biz_type: str) -> Optional[BaseProcessor]:
    """根据业务类型获取对应的文档处理器"""
    return _PROCESSOR_REGISTRY.get(biz_type)
