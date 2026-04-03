# src/ai_engine/core/kb_manager.py
from pathlib import Path
from typing import Dict, Any

import yaml

from ai_engine.core.logger import logger
from ai_engine.core.settings import settings


class KBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KBManager, cls).__new__(cls)
            cls._instance.registry = {}
            cls._instance.default_strategy = {}
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        kb_dir = Path(settings.knowledge_base_dir)

        if not kb_dir.exists():
            logger.warning(f"知识库注册目录不存在: {kb_dir}")
            return

        # 1. 加载默认策略
        default_path = kb_dir / "_default_strategy.yaml"
        if default_path.exists():
            with open(default_path, 'r', encoding='utf-8') as f:
                self.default_strategy = yaml.safe_load(f) or {}

        # 2. 加载所有业务知识库配置
        default_retrieval = self.default_strategy.get("retrieval", {})
        default_context = self.default_strategy.get("context_assembly", {})

        for filepath in kb_dir.glob("*.yaml"):
            # 排除以 _ 开头的文件 (如 _default_strategy.yaml)
            if filepath.name.startswith("_"):
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and "name" in data:
                    # 合并 Retrieval 策略
                    kb_retrieval = data.get("retrieval", {})
                    merged_retrieval = {
                        **default_retrieval,
                        **kb_retrieval
                    }

                    # 合并 Context Assembly 策略
                    kb_context = data.get("context_assembly", {})
                    merged_context = {**default_context, **kb_context}

                    # 重新赋值回 data
                    data["retrieval"] = merged_retrieval
                    data["context_assembly"] = merged_context

                    # 注册到内存字典中
                    self.registry[data["name"]] = data

        logger.info(f"成功加载 {len(self.registry)} 个 KB 业务配置。")

    def get_router_descriptions(self) -> str:
        """为路由大模型生成上下文描述"""
        if not self.registry:
            return "- 暂无可用知识库"

        desc_list = []
        for name, info in self.registry.items():
            desc = f"- {name}: {info.get('description', '')}"
            desc_list.append(desc)
        return "\n".join(desc_list)

    def get_kb_config(self, kb_name: str) -> Dict[str, Any]:
        """获取特定知识库的完整配置"""
        return self.registry.get(kb_name, {})


# 导出一个全局单例
kb_manager = KBManager()
