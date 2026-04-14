from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from langchain_core.documents import Document


class ChatGraphState(TypedDict):
    """State container for the LangGraph chat workflow."""

    # input
    input: str
    history: list

    # routing
    intent: Optional[str]
    biz_type: Optional[str]
    use_tool_agent: Optional[bool]

    # rag pipeline
    rewritten_query: Optional[str]
    documents: List[Document]
    retrieval_score: Optional[float]
    should_retry: Optional[bool]
    context: Optional[str]
    sources: List[str]
    extra_data: Dict[str, Any]
    response_metadata: Dict[str, Any]

    # output
    final_answer: Optional[str]


class ChatGraphInput(TypedDict, total=False):
    """Input schema for the LangGraph chat workflow."""

    input: str
    history: list
    biz_type: Optional[str]
