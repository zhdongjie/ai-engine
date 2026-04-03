from abc import ABC
from pathlib import Path


class BaseProcessor(ABC):

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        extracted_meta = {}
        return text, extracted_meta
