from abc import ABC
from pathlib import Path


class BaseProcessor(ABC):
    def __init__(self, enable_lang_detect: bool = True):
        self.enable_lang_detect = enable_lang_detect

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        extracted_meta = {}

        if self.enable_lang_detect:
            parts = file_path.parts
            if "zh" in parts:
                extracted_meta["lang"] = "zh"
            elif "cht" in parts:
                extracted_meta["lang"] = "cht"
            elif "en" in parts:
                extracted_meta["lang"] = "en"
            else:
                extracted_meta["lang"] = "zh"

        return text, extracted_meta
