from pydantic import BaseModel, Field

from nodes.llm import DEFAULT_MODEL, get_llm
from state import ResearchState


class Report(BaseModel):
    report: str = Field(description="The final research report in markdown")


class Writer:
    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model)
        self.structured_llm = llm.with_structured_output(Report)

    def write(self, state: ResearchState) -> dict:
        print(f"\n[writer] Writing report from {len(state['findings'])} findings")

        findings = "\n\n".join(
            f"Sub question: {f['sub_question']}\nAnswer: {f['answer']}"
            for f in state["findings"]
        )

        prompt = f"""You are the writer in a research agent.
Using only the research findings below, write a clear, well-structured
report in markdown that answers the user's question.

Question: {state["question"]}

Findings:
{findings}
"""
        response = self.structured_llm.invoke(prompt)

        print("[writer] Report done\n")

        return {"report": response.report}
