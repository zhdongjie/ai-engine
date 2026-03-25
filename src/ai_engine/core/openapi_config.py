# src/ai_engine/core/openapi_config.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from ai_engine.core.constants import OPENAPI_TAG_MAPPING, OPENAPI_DEFAULT_TAG
from ai_engine.core.settings import settings


def setup_openapi(app: FastAPI) -> None:
    """
    定制 FastAPI 的 OpenAPI Schema 生成逻辑，优化接口文档展示。
    """

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=settings.PROJECT_NAME,
            version=settings.PROJECT_VERSION,
            description=settings.PROJECT_DESCRIPTION,
            routes=app.routes,
        )

        if "paths" in openapi_schema:
            for path, path_item in openapi_schema["paths"].items():
                for method, operation in path_item.items():
                    # 取原始 tags
                    original_tags = operation.get("tags", [])

                    if not original_tags:
                        # 如果原始 tags 为空，则赋默认标签
                        operation["tags"] = OPENAPI_DEFAULT_TAG
                    else:
                        # 映射 tag，如果 tag 没有在映射中，则保持原样
                        operation["tags"] = [OPENAPI_TAG_MAPPING.get(tag, tag) for tag in original_tags]

        # TODO: 可在此注入 components/securitySchemes
        # openapi_schema["components"]["securitySchemes"] = { ... }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
