import random

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich import print

from langgraph_tutorials.llms import default_llm as llm


def is_user_authenticated() -> bool:
    """Check user authentication."""
    return random.random() > 0.5


def tool_calling_llm(state: MessagesState) -> dict:
    return {"messages": [llm.bind_tools(tools=[is_user_authenticated]).invoke(input=state["messages"])]}


def main():

    builder = StateGraph(MessagesState)

    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools=[is_user_authenticated]))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_edge("tools", END)

    builder.add_conditional_edges(source="tool_calling_llm", path=tools_condition)

    graph = builder.compile()
    print(graph.get_graph().draw_ascii())

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Is the user authenticated?"),
    ]
    response = graph.invoke(input={"messages": messages})
    print(response)


if __name__ == "__main__":
    main()
