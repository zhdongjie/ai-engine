from pathlib import Path

from ai_engine.knowledge.processors.base import BaseProcessor


class DefaultProcessor(BaseProcessor):

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        return super().process(text, file_path)
