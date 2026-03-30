# scripts/processors/base.py

from abc import ABC
from pathlib import Path

class BaseProcessor(ABC):
    """
    文档处理器基类
    """
    def __init__(self, enable_lang_detect: bool = True):
        self.enable_lang_detect = enable_lang_detect

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        """
        基类默认的 process 实现，负责提取通用的元数据（如语言）。
        子类在重写此方法时，应当先调用 super().process()。
        """
        extracted_meta = {}

        # 如果开启了语言探测，基于路径提取语言
        if self.enable_lang_detect:
            parts = file_path.parts
            if "zh" in parts:
                extracted_meta["lang"] = "zh"
            elif "cht" in parts:
                extracted_meta["lang"] = "cht"
            elif "en" in parts:
                extracted_meta["lang"] = "en"
            else:
                extracted_meta["lang"] = "zh"  # 默认兜底简体中文

        # 基类不修改原文，直接返回
        return text, extracted_meta