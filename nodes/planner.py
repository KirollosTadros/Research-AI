from pydantic import BaseModel, Field

from nodes.llm import DEFAULT_MODEL, get_llm
from state import ResearchState


class QuestionBreakdown(BaseModel):
    sub_questions: list[str] = Field(description="List of sub questions")


class Planner:
    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model)
        self.structured_llm = llm.with_structured_output(QuestionBreakdown)

    def plan(self, state: ResearchState) -> dict:
        print(f"\n[planner] Breaking down: {state['question']}")

        prompt = f"""You are the planner in a research agent.
Break the following user question down into focused sub questions
that can each be researched independently.

Question: {state["question"]}
"""
        response = self.structured_llm.invoke(prompt)

        print(f"[planner] Created {len(response.sub_questions)} sub questions:")
        for i, q in enumerate(response.sub_questions, start=1):
            print(f"  {i}. {q}")

        return {"sub_questions": response.sub_questions}
