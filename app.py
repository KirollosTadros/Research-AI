import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from nodes.clarifier import Clarifier, ask_human
from nodes.llm import DEFAULT_MODEL
from nodes.planner import Planner
from nodes.researcher import Researcher
from nodes.writer import Writer
from state import ResearchState


def route_after_clarifier(state: ResearchState) -> str:
    if state["clarifying_question"]:
        return "ask_human"
    if state.get("report"):
        return END
    return "planner"


def dispatch_researchers(state: ResearchState) -> list[Send]:
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


def build_graph(model: str = DEFAULT_MODEL):
    clarifier = Clarifier(model)
    planner = Planner(model)
    researcher = Researcher(model)
    writer = Writer(model)

    graph = StateGraph(ResearchState)
    graph.add_node("clarifier", clarifier.clarify)
    graph.add_node("ask_human", ask_human)
    graph.add_node("planner", planner.plan)
    graph.add_node("researcher", researcher.graph)
    graph.add_node("writer", writer.write)

    graph.add_edge(START, "clarifier")
    graph.add_conditional_edges("clarifier", route_after_clarifier, ["ask_human", "planner", END])
    graph.add_edge("ask_human", "clarifier")
    graph.add_conditional_edges("planner", dispatch_researchers, ["researcher"])
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    user_question = input("Enter what do you want to search for: ")
    result = app.invoke({"question": user_question}, config)

    while "__interrupt__" in result:
        answer = input(f"\n{result['__interrupt__'][0].value}\n> ")
        result = app.invoke(Command(resume=answer), config)

    print(result["report"])
