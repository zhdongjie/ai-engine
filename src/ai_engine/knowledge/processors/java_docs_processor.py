import json
import re
from pathlib import Path

from ai_engine.knowledge.processors.base import BaseProcessor


class JavaDocProcessor(BaseProcessor):
    def __init__(self, enable_lang_detect: bool = False):
        super().__init__(enable_lang_detect=enable_lang_detect)

    def process(self, text: str, file_path: Path) -> tuple[str, dict]:
        content, extracted_meta = super().process(text, file_path)

        start_tag = "<" + "!-- QUESTIONS_START --" + ">"
        end_tag = "<" + "!-- QUESTIONS_END --" + ">"

        if start_tag not in content:
            extracted_meta["questions"] = "[]"
            return content, extracted_meta

        content_main, question_part = content.split(start_tag)
        question_part = question_part.split(end_tag)[0]

        questions = []
        blocks = re.split(r"###\s+", question_part)

        for block in blocks:
            if "URL:" not in block:
                continue
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            title = lines[0]
            url = next((line.replace("URL:", "").strip() for line in lines if line.startswith("URL:")), "")
            if url:
                questions.append({"title": title, "url": url})

        extracted_meta["questions"] = json.dumps(questions, ensure_ascii=False)
        return content_main.strip(), extracted_meta
