import uuid
from enum import StrEnum, auto
from typing import Literal

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from rich import print

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.tracers import langfuse_handler

# main tutorial code


class Location(BaseModel):
    country: str
    city: str


class InterestRoutine(StrEnum):
    MORNING = auto()
    MID_DAY = auto()
    NIGHT = auto()


class Interest(BaseModel):
    name: str
    routine: InterestRoutine


class ContactInfoType(StrEnum):
    PHONE = auto()
    EMAIL = auto()


class ContactInfo(BaseModel):
    contact: str
    type: ContactInfoType


class BaseProfile(BaseModel):
    name: str
    age: int


class Profile(BaseModel):
    base_profile: BaseProfile
    location: Location
    interests: list[Interest]


# prompts
CHATBOT_SYSTEM_PROMPT = """You are a helpful assistant.

Personalize your feedback based on the provided user information.

CURRENT USER INFORMATION:
{memory}
"""

MEMORY_SYSTEM_PROMPT = """You are collecting information about the user to personalize your LLM responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history provided carefully.
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory.
4. Format the memory as a clear, bulleted list.
5. If new information conflicts with existing memory, keep the most recent version.

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences."""

MEMORY_USER_PROMPT = """Update the user information based on the following conversation:

{conversation}
"""


# nodes
def chatbot_node(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:
    user_id: str = config["configurable"]["user_id"]

    name_space = ("memory", user_id)
    key = "user_memory"
    user_memory = store.get(namespace=name_space, key=key)

    memory = ""
    if user_memory:
        memory = user_memory.value.get("memory")

    system_message = CHATBOT_SYSTEM_PROMPT.format(memory=memory)

    response: AIMessage = llm.invoke(input=[SystemMessage(content=system_message)] + state["messages"])

    return {"messages": [AIMessage(content=response.text)]}


def memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore) -> None:
    user_id: str = config["configurable"]["user_id"]

    name_space = ("memory", user_id)
    key = "user_memory"
    user_memory = store.get(namespace=name_space, key=key)

    memory = ""
    if user_memory:
        memory = user_memory.value.get("memory")

    conversation = "\n".join(f"{msg.type}: {msg.text}" for msg in state["messages"])

    system_message = MEMORY_SYSTEM_PROMPT.format(memory=memory)
    user_message = MEMORY_USER_PROMPT.format(conversation=conversation)

    updated_profile: Profile = llm.with_structured_output(schema=Profile).invoke(
        input=[SystemMessage(content=system_message), HumanMessage(content=user_message)]
    )

    store.put(namespace=name_space, key=key, value={"memory": updated_profile.model_dump(mode="json")})


# graph
builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("memory", memory_node)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "memory")
builder.add_edge("memory", END)

memory_store = InMemoryStore()
memory_saver = InMemorySaver()

graph = builder.compile(checkpointer=memory_saver, store=memory_store, name="chatbot-with-user-memory")

with open("graph.md", "w+", encoding="utf-8") as f:
    f.write(f"""```mermaid\n{graph.get_graph(xray=True).draw_mermaid(with_styles=False)}\n```""")


def main():
    thread_id = uuid.uuid4().hex
    user_id = "hossam"
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
        },
    }

    while True:
        user_input = input("Human: ")
        if user_input == "break":
            break

        user_message = HumanMessage(content=user_input)

        for chunk in graph.stream({"messages": [user_message]}, config, stream_mode="updates"):
            print(chunk)

    print(graph.store.list_namespaces())
    print(graph.store.get(namespace=graph.store.list_namespaces()[0], key="user_memory"))


if __name__ == "__main__":
    main()
