# src/ai_engine/chains/rag_plugins/i18n_plugin.py
from typing import Dict, Any, List

from langchain_core.runnables import RunnableConfig

from ai_engine.chains.rag_plugins.base import BaseRAGPlugin


class I18nInstructionPlugin(BaseRAGPlugin):
    def process(
            self,
            docs: List[Any],
            context: str,
            extra: Dict[str, Any],
            config: RunnableConfig
    ):
        configurable = config.get("configurable") or {}
        lang = configurable.get("lang", "zh")

        instructions = {
            "en": "Please respond in English based on the context above.",
            "cht": "請務必使用繁體中文回答上述内容。",
        }

        if patch := instructions.get(lang):
            extra.setdefault("injected_messages", [])
            extra["injected_messages"].append(("system", patch))

        return context, extra
