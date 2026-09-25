from pydantic import BaseModel, Field

from nodes.llm import DEFAULT_MODEL, get_llm
from state import ResearcherState


class Finding(BaseModel):
    answer: str = Field(description="A concise, factual answer to the sub question")


class Researcher:
    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model)
        self.structured_llm = llm.with_structured_output(Finding)

    def research(self, state: ResearcherState) -> dict:
        sub_question = state["sub_question"]
        print(f"[researcher] Started: {sub_question}")

        prompt = f"""You are a researcher in a research agent.
Answer the following sub question accurately and concisely.
Only state facts you are confident about. If the sub question is too vague
to answer factually, or you don't know the answer, say so plainly instead
of guessing. Never invent facts.

Sub question: {sub_question}
"""
        response = self.structured_llm.invoke(prompt)

        print(f"[researcher] Finished: {sub_question}")

        return {"findings": [{"sub_question": sub_question, "answer": response.answer}]}
