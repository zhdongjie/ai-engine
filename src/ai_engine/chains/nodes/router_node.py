# src/ai_engine/chains/nodes/router_node.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from ai_engine.chains.nodes.normal_node import normal_chat_run
from ai_engine.chains.nodes.rag_node import dynamic_rag_run
from ai_engine.core.logger import logger
from ai_engine.core.prompt_manager import get_prompt_config
from ai_engine.infra.llm.llm_factory import LLMFactory

router_config = get_prompt_config("intent_router")

router_llm = LLMFactory.get_model(router_config.get("config", {}))

router_prompt = PromptTemplate.from_template(router_config["content"])
intent_router = router_prompt | router_llm | StrOutputParser()


def route_logger(info: dict) -> dict:
    intent = info.get("intent", "UNKNOWN").strip().upper()
    logger.info(f"路由判定结果: [{intent}] | 用户输入: {info.get('input', '')[:15]}...")
    return info


def route_logic(info: dict):
    if "RAG" in info.get("intent", "").upper():
        return RunnableLambda(dynamic_rag_run)
    return RunnableLambda(normal_chat_run)
