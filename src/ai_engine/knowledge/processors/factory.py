from typing import Optional

from ai_engine.knowledge.processors.base import BaseProcessor
from ai_engine.knowledge.processors.default_processor import DefaultProcessor
from ai_engine.knowledge.processors.java_docs_processor import JavaDocProcessor

_PROCESSOR_REGISTRY = {
    "java_tutor": JavaDocProcessor(),
}


def get_processor(biz_type: str) -> Optional[BaseProcessor]:
    return _PROCESSOR_REGISTRY.get(biz_type, DefaultProcessor())
