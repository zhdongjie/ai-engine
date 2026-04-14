from typing import Any, Dict

from ai_engine.core.prompt_manager import get_prompt_config


def load_prompt(name: str) -> Dict[str, Any]:
    """Load a prompt configuration by name."""
    return get_prompt_config(name)
