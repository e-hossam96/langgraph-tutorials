from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rich import print

from langgraph_tutorials.tracers import langfuse_handler


class State(TypedDict):
    input: str


def node_1(state: State) -> State:
    print("--node_1--")
    return state


def node_2(state: State) -> State:
    if len(state["input"]) > 4:
        raise interrupt(value=f"Input is longer than 4 chars: {state['input']}")
    print("--node_2--")
    return state


def node_3(state: State) -> State:
    print("--node_3--")
    return state


def main():
    builder = StateGraph(State)

    # nodes
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    # edges
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", "node_3")
    builder.add_edge("node_3", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    with open("graph.mmd", "w+", encoding="utf-8") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid())

    thread_id = "1"
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
        },
    }
    for event in graph.stream(input=State(input="Hello"), config=config, stream_mode="values"):
        print(event)
        print("-" * 40)

    state = graph.get_state(config=config)
    print(state)
    print("=" * 40)
    graph.update_state(config=config, values=State(input="Hi"))

    state = graph.get_state(config=config)
    print(state)
    print("=" * 40)
    for event in graph.stream(input=None, config=config, stream_mode="values"):
        print(event)
        print("-" * 40)


if __name__ == "__main__":
    main()
