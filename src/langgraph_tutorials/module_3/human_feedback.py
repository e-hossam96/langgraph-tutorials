import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich import print

from langgraph_tutorials.llms import default_llm as llm


def is_user_authenticated() -> bool:
    """Check user authentication."""
    return random.random() > 0.5


def human_feedback(_: MessagesState) -> None: ...


def tool_calling_llm(state: MessagesState) -> MessagesState:
    llm_response: AIMessage = llm.bind_tools(tools=[is_user_authenticated]).invoke(input=state["messages"])
    if llm_response.content[-1]["type"] == "function_call":
        state = MessagesState(messages=[AIMessage(content="", tool_calls=llm_response.tool_calls)])
    else:
        state = MessagesState(messages=[AIMessage(content=llm_response.content[-1]["text"])])
    return state


async def main():

    builder = StateGraph(MessagesState)

    builder.add_node("human_feedback", human_feedback)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools=[is_user_authenticated]))

    builder.add_edge(START, "human_feedback")
    builder.add_edge("human_feedback", "tool_calling_llm")
    builder.add_edge("tools", "human_feedback")

    builder.add_conditional_edges(source="tool_calling_llm", path=tools_condition)

    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["human_feedback"])
    with open("graph.mmd", "w+", encoding="utf-8") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid())

    config = {"configurable": {"thread_id": "1"}}
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Is the user authenticated?"),
    ]
    response = graph.invoke(input={"messages": messages}, config=config)

    graph.update_state(
        config=config,
        as_node="human_feedback",
        values=MessagesState(
            messages=[HumanMessage(content="Actually, tell me who you are first before checking authentication.")]
        ),
    )

    response = graph.invoke(input=None, config=config)
    response = graph.invoke(input=None, config=config)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
