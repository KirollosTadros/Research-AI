import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from tavily import TavilyClient

from nodes.llm import DEFAULT_MODEL, get_llm
from state import ResearcherAgentState, ResearcherOutput, ResearcherState

MAX_SEARCH_ROUNDS = 3

SYSTEM_PROMPT = """You are a researcher in a research agent.
Answer the user's sub question accurately and concisely.
Use the web_search tool whenever the answer depends on facts, numbers,
or recent information. You can search more than once to refine your query.
Only state facts supported by your search results or that you are confident about.
If you can't find the answer, say so plainly instead of guessing. Never invent facts.
End your answer with a "Sources:" list of the URLs you used.
Only list URLs that appear in your search results."""


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information. Returns the top results with their URLs."""
    print(f"[web_search] {query}")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(query, max_results=5)

    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in response["results"]
    )


class Researcher:
    def __init__(self, model: str = DEFAULT_MODEL):
        llm = get_llm(model)
        self.llm_with_tools = llm.bind_tools([web_search])
        self.llm_answer_only = llm.bind_tools([web_search], tool_choice="none")

        graph = StateGraph(ResearcherAgentState, input_schema=ResearcherState, output_schema=ResearcherOutput)
        graph.add_node("agent", self.agent)
        graph.add_node("tools", ToolNode([web_search]))
        graph.add_node("finish", self.finish)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "finish"})
        graph.add_edge("tools", "agent")
        graph.add_edge("finish", END)

        self.graph = graph.compile()

    def agent(self, state: ResearcherAgentState) -> dict:
        new_messages = []
        if not state.get("messages"):
            print(f"[researcher] Started: {state['sub_question']}")
            new_messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["sub_question"])]

        messages = state.get("messages", []) + new_messages
        search_rounds = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)

        if search_rounds < MAX_SEARCH_ROUNDS:
            response = self.llm_with_tools.invoke(messages)
        else:
            new_messages.append(HumanMessage("Search limit reached. Answer now using only the search results above."))
            response = self.llm_answer_only.invoke(messages + new_messages[-1:])
            if response.tool_calls:
                response = AIMessage(response.text or "I couldn't find a reliable answer within the search limit.")

        return {"messages": new_messages + [response]}

    def finish(self, state: ResearcherAgentState) -> dict:
        sub_question = state["sub_question"]
        print(f"[researcher] Finished: {sub_question}")

        return {"findings": [{"sub_question": sub_question, "answer": state["messages"][-1].text}]}
