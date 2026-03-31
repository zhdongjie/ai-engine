# src/ai_engine/chains/interceptors/i18n.py
from typing import Dict, Any

def i18n_input_interceptor(info: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """
    输入拦截器：写入 metadata['lang']，不改内容
    """
    configurable = getattr(config, "get", lambda x, default=None: {})("configurable") or {}
    user_lang = configurable.get("lang", "zh")

    metadata = info.get("__metadata__", {})
    metadata["lang"] = user_lang
    info["__metadata__"] = metadata

    return info