from typing import TypedDict, Annotated
import operator

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
