from langgraph.types import interrupt
from pydantic import BaseModel, Field

from nodes.llm import DEFAULT_MODEL, get_llm
from state import ResearchState

MAX_CLARIFICATION_ROUNDS = 2


class Clarification(BaseModel):
    needs_clarification: bool = Field(
        description="True only if the question is too vague or ambiguous to research well"
    )
    clarifying_question: str = Field(
        description="One short question to ask the user, or an empty string if not needed"
    )


class Clarifier:
    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model)
        self.structured_llm = llm.with_structured_output(Clarification)

    def clarify(self, state: ResearchState) -> dict:
        rounds = state.get("clarification_rounds", 0)
        print(f"\n[clarifier] Checking: {state['question']}")

        prompt = f"""You are the clarifier in a research agent.
Decide if the user's research request is clear enough to research.
Only ask for clarification if the request is genuinely vague or ambiguous,
or important information is missing. Do not ask about minor details.
Placeholder names like x and y, or words like "it" or "which" with nothing
they refer to, do not count as a specific topic.

Request: {state["question"]}
"""
        response = self.structured_llm.invoke(prompt)

        if not response.needs_clarification:
            print("[clarifier] Question is clear")
            return {"clarifying_question": ""}

        if rounds >= MAX_CLARIFICATION_ROUNDS:
            print("[clarifier] Still unclear after max rounds, giving up")
            return {
                "clarifying_question": "",
                "report": "I still couldn't work out what you'd like me to research. "
                "Please ask again with the specific topic or options you have in mind.",
            }

        print(f"[clarifier] Needs clarification: {response.clarifying_question}")
        return {
            "clarifying_question": response.clarifying_question,
            "clarification_rounds": rounds + 1,
        }


def ask_human(state: ResearchState) -> dict:
    answer = interrupt(state["clarifying_question"])

    print(f"[ask_human] User answered: {answer}")
    return {
        "question": f"{state['question']}\n"
        f"Clarification - Q: {state['clarifying_question']} A: {answer}"
    }
