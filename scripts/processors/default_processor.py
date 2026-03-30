# scripts/processors/default_processor.py
from pathlib import Path

from .base import BaseProcessor


class DefaultProcessor(BaseProcessor):

    def __init__(self, enable_lang_detect: bool = True):
        super().__init__(enable_lang_detect=enable_lang_detect)

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        return super().process(text, file_path)
