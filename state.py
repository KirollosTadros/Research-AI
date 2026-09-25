from typing import TypedDict, Annotated
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[dict], operator.add]
    iterations: int
    report: str
    clarifying_question: str
    clarification_rounds: int

class ResearcherState(TypedDict):
    sub_question: str

class ResearcherOutput(TypedDict):
    findings: Annotated[list[dict], operator.add]

class ResearcherAgentState(ResearcherState, ResearcherOutput):
    messages: Annotated[list[AnyMessage], add_messages]
