from typing import List

from ai_engine.core.kb_manager import kb_manager


def resolve_multi_rag_targets(biz_type: str) -> List[str]:
    """Resolve additional KB targets for multi-RAG from KB config."""
    kb_config = kb_manager.get_kb_config(biz_type)
    raw_config = kb_config.get("multi_rag") or kb_config.get("multi_rag_targets")

    targets: List[str] = []
    if isinstance(raw_config, list):
        targets = [str(item).strip() for item in raw_config]
    elif isinstance(raw_config, dict):
        raw_targets = raw_config.get("targets") or []
        if isinstance(raw_targets, list):
            targets = [str(item).strip() for item in raw_targets]

    deduped = []
    for target in targets:
        if not target or target == biz_type:
            continue
        if target not in kb_manager.registry:
            continue
        if target not in deduped:
            deduped.append(target)

    return deduped
