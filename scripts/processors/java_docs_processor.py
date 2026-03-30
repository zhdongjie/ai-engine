# scripts/processors/java_docs_processor.py
import re
import json
from pathlib import Path
from .base import BaseProcessor


class JavaDocProcessor(BaseProcessor):

    def __init__(self, enable_lang_detect: bool = False):
        super().__init__(enable_lang_detect=enable_lang_detect)

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        content, extracted_meta = super().process(text, file_path)

        START_TAG = "<" + "!-- QUESTIONS_START --" + ">"
        END_TAG = "<" + "!-- QUESTIONS_END --" + ">"

        if START_TAG not in content:
            extracted_meta["questions"] = "[]"
            return content, extracted_meta

        content_main, q_part = content.split(START_TAG)
        q_part = q_part.split(END_TAG)[0]

        questions = []
        blocks = re.split(r'###\s+', q_part)

        for b in blocks:
            if "URL:" not in b: continue
            lines = [l.strip() for l in b.split("\n") if l.strip()]
            title = lines[0]
            url = next((l.replace("URL:", "").strip() for l in lines if l.startswith("URL:")), "")
            if url:
                questions.append({"title": title, "url": url})

        extracted_meta["questions"] = json.dumps(questions, ensure_ascii=False)

        return content_main.strip(), extracted_meta