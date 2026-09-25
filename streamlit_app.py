import uuid

import streamlit as st
from langgraph.types import Command

from app import build_graph
from nodes.llm import AVAILABLE_MODELS, DEFAULT_MODEL


@st.cache_resource
def get_graph(model: str):
    return build_graph(model)


def show_message(message: dict):
    with st.chat_message(message["role"]):
        if message.get("steps"):
            with st.expander("Research steps"):
                st.markdown("\n\n".join(message["steps"]))
        st.markdown(message["content"])


def run_graph(graph_input):
    graph = get_graph(st.session_state.model)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    steps = []
    reply = "Something went wrong, please try again."

    with st.chat_message("assistant"):
        with st.status("Working...", expanded=True) as status:
            for update in graph.stream(graph_input, config, stream_mode="updates"):
                for node, output in update.items():
                    step = None
                    if node == "__interrupt__":
                        reply = output[0].value
                        st.session_state.waiting_for_answer = True
                    elif node == "clarifier":
                        if output["clarifying_question"]:
                            step = "**Clarifier** needs more information"
                        elif output.get("report"):
                            reply = output["report"]
                            step = "**Clarifier** couldn't understand the question"
                        else:
                            step = "**Clarifier** says the question is clear"
                    elif node == "planner":
                        sub_questions = "\n".join(f"- {q}" for q in output["sub_questions"])
                        step = f"**Planner** created {len(output['sub_questions'])} sub questions:\n\n{sub_questions}"
                    elif node == "researcher":
                        step = f"**Researcher** finished: {output['findings'][0]['sub_question']}"
                    elif node == "writer":
                        reply = output["report"]
                        step = "**Writer** finished the report"

                    if step:
                        st.markdown(step)
                        steps.append(step)

            status.update(label="Research steps", state="complete", expanded=False)

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply, "steps": steps})


st.set_page_config(page_title="Research Agent")
st.title("Research Agent")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("waiting_for_answer", False)

with st.sidebar:
    model = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.waiting_for_answer = False
        st.rerun()

for message in st.session_state.messages:
    show_message(message)

placeholder = "Answer the question above" if st.session_state.waiting_for_answer else "What do you want to research?"

if prompt := st.chat_input(placeholder):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    if st.session_state.waiting_for_answer:
        st.session_state.waiting_for_answer = False
        run_graph(Command(resume=prompt))
    else:
        st.session_state.model = model
        st.session_state.thread_id = str(uuid.uuid4())
        run_graph({"question": prompt})

    st.rerun()
