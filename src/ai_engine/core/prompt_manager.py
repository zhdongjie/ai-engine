# src/ai_engine/core/prompt_manager.py
import os
from typing import Any, Dict

import yaml

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings


def _read_prompt_file(prompt_name: str) -> Dict[str, Any] | None:
    """
    内部私有函数：负责具体的寻址和读取逻辑
    支持 .yaml -> .yml -> .txt
    """
    prompt_dir = settings.get_prompt_path("")
    extensions = [".yaml", ".yml", ".txt"]

    for ext in extensions:
        file_path = os.path.join(prompt_dir, f"{prompt_name}{ext}")
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # 1. 处理 YAML/YML
                if ext in [".yaml", ".yml"]:
                    data = yaml.safe_load(f) or {}
                    # 规范化输出：确保包含 content 和 config 键
                    return {
                        "content": data.get("content", "").strip(),
                        "config": data.get("config", {}),
                        "source": file_path
                    }

                # 2. 处理纯文本 TXT (TXT 没有 config，给个空的)
                else:
                    content = f.read().strip()
                    return {
                        "content": content,
                        "config": {},
                        "source": file_path
                    }
        except Exception as e:
            logger.error(f"❌ 读取 Prompt 文件 {file_path} 出错: {e}")
            return None
    return None


def get_prompt_config(prompt_name: str = "default") -> Dict[str, Any]:
    """
    核心入口：获取 Prompt 数据，带自动兜底逻辑
    返回格式: {"content": str, "config": dict}
    """
    # 1. 尝试加载请求的特定业务 Prompt
    result = _read_prompt_file(prompt_name)

    if result:
        logger.debug(f"🎯 成功加载业务 Prompt: {prompt_name} (来自 {result['source']})")
        return result

    # 2. 如果加载失败且当前不是 default，尝试加载 default 兜底
    if prompt_name != "default":
        logger.warning(f"⚠️ 未找到业务 Prompt '{prompt_name}'，尝试加载母版 'default'...")
        default_result = _read_prompt_file("default")
        if default_result:
            return default_result

    # 3. 终极防御：如果连 default 都没有，返回代码硬编码的最小化指令
    logger.error("🚨 严重警告：未找到任何 Prompt 文件（包括 default），使用系统硬编码兜底！")
    return {
        "content": "你是一个专业的 AI 助手。请根据已知知识回答问题：\n\n{context}",
        "config": {"temperature": 0, "model": settings.QWEN_MODEL_LLM}
    }
