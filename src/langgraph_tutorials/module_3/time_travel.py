import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich import print

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.tracers import langfuse_handler


def is_user_authenticated() -> bool:
    """Check user authentication."""
    return random.random() > 0.5


def tool_calling_llm(state: MessagesState) -> MessagesState:
    llm_response: AIMessage = llm.bind_tools(tools=[is_user_authenticated]).invoke(input=state["messages"])
    if llm_response.content[-1]["type"] == "function_call":
        state = MessagesState(messages=[AIMessage(content="", tool_calls=llm_response.tool_calls)])
    else:
        state = MessagesState(messages=[AIMessage(content=llm_response.content[-1]["text"])])
    return state


async def main():
    builder = StateGraph(MessagesState)

    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools=[is_user_authenticated]))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_edge("tools", "tool_calling_llm")
    builder.add_conditional_edges(source="tool_calling_llm", path=tools_condition)

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
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Is the user authenticated?"),
    ]
    response = graph.invoke(input={"messages": messages}, config=config)
    print(response)
    print("=" * 40)

    all_states = [s for s in graph.get_state_history(config=config)]
    print(f"Num states: {len(all_states)}")
    print("=" * 40)

    # replay state checkpoint
    to_replay = all_states[-3]  # tool call
    print(to_replay.config)
    print("=" * 40)

    async for event in graph.astream(input=None, config=to_replay.config, stream_mode="updates"):
        print(event)
        print("-" * 40)

    # fork from state checkpoint
    to_fork = all_states[-2]  # llm call
    print(to_fork.config)
    print("=" * 40)

    fork_config = graph.update_state(
        config=to_fork.config,
        # as_node="tool_calling_llm",
        values=MessagesState(messages=[HumanMessage(content="Who are you?", id=to_fork.values["messages"][-1].id)]),
    )

    async for event in graph.astream(input=None, config=fork_config, stream_mode="updates"):
        print(event)
        print("-" * 40)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
