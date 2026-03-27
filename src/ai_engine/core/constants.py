# src/ai_engine/core/constants.py

# LangServe/OpenAPI 标签映射
OPENAPI_TAG_MAPPING = {
    "default": "LangSmith 监控与追踪",
    "chat/config": "LangServe 沙盒配置",
    "chat": "AI 核心对话流",
    "invoke": "AI 对话接口",
    "stream": "AI 流式接口",
    "batch": "批量调用接口"
}

# 默认 tags
OPENAPI_DEFAULT_TAG = ["LangServe 默认接口"]

# LangServe白名单
LANG_SERVE_ALLOWED_ENDPOINTS = ["invoke", "stream"]
