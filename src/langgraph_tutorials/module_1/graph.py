import random
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from rich import print


class State(TypedDict):
    graph_state: str


# from dataclasses import dataclass


# @dataclass
# class State:
#     graph_state: str


def node_1(state: State) -> State:
    print("--node_1--")
    return State(graph_state=state["graph_state"] + " I'm")


def node_2(state: State) -> State:
    print("--node_2--")
    return State(graph_state=state["graph_state"] + " happy.")


def node_3(state: State) -> State:
    print("--node_3--")
    return State(graph_state=state["graph_state"] + " sad.")


def choose_node(_: State) -> Literal["node_2", "node_3"]:
    if random.random() > 0.5:
        return "node_2"
    return "node_3"


def main():
    builder = StateGraph(State)

    # nodes
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    # edges
    builder.add_edge(START, "node_1")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    builder.add_conditional_edges("node_1", choose_node)

    graph = builder.compile()
    print(graph.get_graph().draw_ascii())
    result = graph.invoke(input=State(graph_state="Hello, this is Hossam."))
    print(result)


if __name__ == "__main__":
    main()
