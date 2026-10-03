from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from rich import print

from langgraph_tutorials.llms import default_llm as llm


class SummarizationMessageState(MessagesState):
    summary: str | None


def summarize_conversation(state: SummarizationMessageState) -> SummarizationMessageState:
    summary = state.get("summary")

    if summary is not None:
        summary_message = HumanMessage(
            content=(
                f"Here is a summary of an earlier state of the conversation\n{summary}\n"
                "Extend this summary by summarizing the current conversation state above."
            )
        )
    else:
        summary_message = HumanMessage(content="Create a summary of the conversation above.")

    messages = state["messages"] + [summary_message]
    response: AIMessage = llm.invoke(input=messages)

    # delete previous messages except last two
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return SummarizationMessageState(summary=response.content[-1]["text"], messages=delete_messages)


def call_llm(state: SummarizationMessageState) -> SummarizationMessageState:
    summary = state.get("summary")

    if summary is not None:
        system_message = SystemMessage(content=f"Here is a summary of the conversation\n{summary}")
        messages = [system_message] + state["messages"]
    else:
        messages = state["messages"]

    response: AIMessage = llm.invoke(input=messages)
    return SummarizationMessageState(summary=summary, messages=[AIMessage(content=response.content[-1]["text"])])


def should_summarize(state: SummarizationMessageState) -> Literal["summarization", END]:
    num_messages = len(state["messages"])
    return "summarization" if num_messages > 6 else END


async def main():

    builder = StateGraph(MessagesState)

    builder.add_node("conversation", call_llm)
    builder.add_node("summarization", summarize_conversation)

    builder.add_edge(START, "conversation")
    builder.add_edge("summarization", END)

    builder.add_conditional_edges(source="conversation", path=should_summarize)

    graph = builder.compile(checkpointer=InMemorySaver())
    print(graph.get_graph().draw_ascii())

    config = {"configurable": {"thread_id": "1"}}

    # turn 1
    messages = [
        SystemMessage(content=("You are a helpful assistant. Your responses are no more than 20 words.")),
        HumanMessage(content="Hello, I'm Hossam"),
        AIMessage(content="Hey Hossam. How can I help you today."),
        HumanMessage(content="Can you write a simple print statement in python?"),
        AIMessage(content="Yes sure. ```print('Hi')```"),
        HumanMessage(content="Do you know anything about python's oop?"),
    ]
    async for update in graph.astream_events(input={"messages": messages}, config=config, version="v2"):
        event = update["event"]
        name = update["name"]
        node = update["metadata"].get("langgraph_node", "******")
        data = update["data"]
        if event != "on_chat_model_stream" or node != "conversation":
            continue
        # print(f"Event: {event} | Name: {name} | Node: {node}")
        # print(f"Data: {data}")
        if data["chunk"].content and data["chunk"].content[0]["type"] == "text":
            print(data["chunk"].content[0]["text"], end=" ")
        # print("=" * 40)

    # print(graph.get_state(config=config))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
