import datetime
import json
import uuid
from enum import StrEnum, auto
from typing import Literal

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from rich import print

from langgraph_tutorials.llms import default_llm
from langgraph_tutorials.tracers import langfuse_handler


class Location(BaseModel):
    country: str | None = None
    city: str | None = None
    is_current: bool | None = None


class ConnectionType(StrEnum):
    FAMILY = auto()
    FRIEND = auto()
    COLLEAGUE = auto()
    OTHER = auto()


class Connection(BaseModel):
    connection_type: ConnectionType | None = None
    name: str | None = None


class Profile(BaseModel):
    id: uuid.UUID
    name: str | None = None
    job: str | None = None
    location: Location | None = None
    connections: list[Connection] = Field(default_factory=lambda: list[Connection]())
    interests: list[str] = Field(default_factory=lambda: list[str]())


class ProfileUpdate(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str | None = None
    job: str | None = None
    location: Location | None = None
    connections: list[Connection] = Field(default_factory=lambda: list[Connection]())
    interests: list[str] = Field(default_factory=lambda: list[str]())


class Status(StrEnum):
    PENDING = auto()
    ACTIVE = auto()
    DONE = auto()


class ToDo(BaseModel):
    id: uuid.UUID
    task: str | None = None
    deadline: datetime.datetime | None = None
    status: Status | None = None


class ToDoUpdate(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task: str | None = None
    deadline: datetime.datetime | None = None
    status: Status | None = None


class Instruction(BaseModel):
    id: uuid.UUID
    instruction: str | None = None


class InstructionUpdate(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    instruction: str | None = None


class UpdateType(StrEnum):
    PROFILE = auto()
    TODO = auto()
    INSTRUCTION = auto()
    RESPOND = auto()


class Update(BaseModel):
    update_type: UpdateType


# maestro node

MAESTRO_SYSTEM_PROMPT = """# ROLE
You are the memory router of a ToDo companion assistant.
You do NOT talk to the user. Your only job is to choose ONE value for `update_type`.

# LONG-TERM MEMORY
The assistant remembers three kinds of information about the user.

1. User Profile: facts about the user (name, job, location, family, friends, colleagues, interests).
```json
{user_profile}
```

2. ToDo List: tasks the user needs to do, with optional deadline and status.
```json
{user_todos}
```

3. Instructions: the user's preferences for HOW the ToDo list should be managed
(for example: "always add a deadline", "group shopping items into one task").
```json
{user_instructions}
```

An empty section means nothing is saved yet.

# ALLOWED VALUES FOR `update_type`
- `profile`     -> the user shared NEW or CHANGED personal facts that are not already in the User Profile.
- `todo`        -> the user mentioned a NEW task, or asked to change / complete / reschedule an existing task.
- `instruction` -> the user stated a NEW or CHANGED preference about how to manage the ToDo list.
- `none`        -> nothing needs to be saved, or everything has already been saved. The assistant will now reply.

# HOW TO DECIDE (follow these steps in order)
Step 1. Find the most recent message from the human. Focus on it.
Step 2. Look at the AI messages that come AFTER it. They are internal logs:
        - A JSON message like "update_type": "todo" is a previous routing decision.
        - A message starting with "New profile created", "Existing profile updated",
          "Updates for existing ToDos" or "Updates for existing Instructions" means that save is DONE.
Step 3. Make a list of every memory type the latest human message needs saved.
Step 4. Remove from that list every type that is already DONE (Step 2) or already present in memory above.
Step 5. If the list is empty, answer `none`. Otherwise answer the FIRST remaining type in this order:
        `profile`, then `todo`, then `instruction`.

# RULES
- Never choose the same type twice for the same human message. If it was already saved, move on.
- Only save what the user actually said. Do not guess or invent facts.
- Greetings, thanks, small talk and questions (for example "what is on my list?") need `none`.
- A preference about the list itself ("remind me to...", "always...", "never...") is an `instruction`,
  not a `todo`.
- When you are unsure, choose `none`.

# EXAMPLES
- "Hi, I'm Sara, a nurse living in Cairo."                -> `profile`
- "I need to buy milk tomorrow."                         -> `todo`
- "I finished the report."                               -> `todo` (mark it done)
- "Please always give my tasks a deadline."              -> `instruction`
- "Thanks!"                                              -> `none`
- Latest AI message says "Updates for existing ToDos..." and nothing else is needed -> `none`
"""


def maestro(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:
    user_id = config["configurable"]["user_id"]
    profile_namespace = (UpdateType.PROFILE, user_id)
    todo_namespace = (UpdateType.TODO, user_id)
    instruction_namespace = (UpdateType.INSTRUCTION, user_id)

    profiles = store.search(profile_namespace, limit=1)
    user_profile = ""
    if profiles:
        user_profile = profiles[0].value

    todos = store.search(todo_namespace)
    user_todos = ""
    if todos:
        user_todos = "\n".join(f"{todo.value}" for todo in todos)

    instructions = store.search(instruction_namespace)
    user_instructions = ""
    if instructions:
        user_instructions = "\n".join(f"{instruction.value}" for instruction in instructions)

    system_message = MAESTRO_SYSTEM_PROMPT.format(
        user_profile=user_profile, user_todos=user_todos, user_instructions=user_instructions
    )
    update: Update = default_llm.with_structured_output(schema=Update).invoke(
        input=[SystemMessage(content=system_message)] + state["messages"]
    )

    return {"messages": [AIMessage(content=f"{json.dumps(update.model_dump(mode='json'))}")]}


# update profile node

UPDATE_PROFILE_SYSTEM_PROMPT = """# ROLE
You extract facts about the user from a conversation and return them as a profile update.

# INPUT
You receive a conversation between a human and an AI assistant.
Some AI messages are internal logs, for example "New profile created. Profile: ..." or
"Existing profile updated. Updates: ...". These logs show the profile that is already saved, including its `id`.

# FIELDS
- `id`: the uuid of the saved profile.
- `name`: the user's name.
- `job`: the user's job or occupation.
- `location`: `country`, `city`, and `is_current` (true if the user lives there now).
- `connections`: people the user knows. Each has a `name` and a `connection_type`:
  `family`, `friend`, `colleague`, or `other`.
- `interests`: short phrases for hobbies and things the user likes (for example "hiking", "jazz").

# STEPS
Step 1. Search the AI log messages for an existing profile `id`.
        - If you find one, copy it EXACTLY, character by character. This updates the existing profile.
        - If you do not find one, leave `id` empty. A new profile will be created.
Step 2. Read what the human said and collect facts about the human themselves.
Step 3. Fill in only the fields you have information for. Leave the other fields empty.

# RULES
- Only use facts the human actually stated. Never guess or invent.
- Facts about the human only. A task ("I need to call mom") is not a profile fact,
  but "mom" can be a `family` connection.
- List fields REPLACE the saved list. When you add to `connections` or `interests`,
  return the FULL list: every saved item plus the new ones.
- If a fact changed (for example the user moved city), return the new value.
- Never make up a uuid. Either copy an existing one exactly or leave it empty.
"""


def update_profile(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:

    profile_update: ProfileUpdate = default_llm.with_structured_output(schema=ProfileUpdate).invoke(
        input=[SystemMessage(content=UPDATE_PROFILE_SYSTEM_PROMPT)] + state["messages"][:-1]
    )

    user_id = config["configurable"]["user_id"]
    profile_namespace = (UpdateType.PROFILE, user_id)
    user_profile = store.get(namespace=profile_namespace, key=profile_update.id.hex)

    if user_profile is None:  # new profile
        profile = Profile.model_validate({**profile_update.model_dump(exclude_none=True)})
        message = f"New profile created.\n\nProfile:\n{profile.model_dump(mode='json')}"
    else:  # existing profile
        profile = Profile.model_validate(user_profile.value).model_copy(
            update={**profile_update.model_dump(exclude_none=True)}
        )
        message = f"Existing profile updated.\n\nUpdates:\n{profile.model_dump(mode='json', exclude_none=True)}"

    store.put(namespace=profile_namespace, key=profile.id.hex, value=profile.model_dump(mode="json"))
    return {"messages": [AIMessage(content=message)]}


# update todos node

UPDATE_TODOS_SYSTEM_PROMPT = """# ROLE
You extract ToDo items from a conversation. You create new tasks and update existing ones.

# INPUT
You receive a conversation between a human and an AI assistant.
Some AI messages are internal logs, for example "Updates for existing ToDos: ..." and "Created ToDos: ...".
These logs show tasks that are already saved, including their `id`.

# FIELDS OF EACH ToDo
- `id`: the uuid of an existing task. Leave empty for a new task.
- `task`: a short, clear description starting with a verb (for example "Buy milk", "Call the dentist").
- `deadline`: a date-time in ISO 8601 format (for example "2026-09-30T17:00:00"). Leave empty if none was given.
- `status`: one of `pending`, `active`, `done`.
  Use `pending` for new tasks, `active` if the user says they started it, `done` if they finished it.

# STEPS
Step 1. Read the latest human messages and find every task they mention.
Step 2. For each task, check the AI log messages for the SAME task that is already saved.
        - Same task found  -> copy its `id` EXACTLY and return only the fields that change
          (for example `status` = `done`).
        - Not found        -> leave `id` empty and fill in `task`, `deadline` if known, and `status` = `pending`.
Step 3. Return all new and changed tasks in `todo_list`.

# RULES
- One ToDo per task. If the user lists three things, return three items.
- Do NOT return tasks that did not change.
- Do NOT create a duplicate of a task that already exists. Update it instead.
- Only use tasks the human actually mentioned. Never invent tasks, deadlines or ids.
- Never make up a uuid. Either copy an existing one exactly or leave it empty.
- If nothing needs to be saved, return an empty `todo_list`.
"""


def update_todos(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:

    class ToDoList(BaseModel):
        todo_list: list[ToDoUpdate] = Field(default_factory=lambda: list[ToDoUpdate]())

    todo_list: ToDoList = default_llm.with_structured_output(schema=ToDoList).invoke(
        input=[SystemMessage(content=UPDATE_TODOS_SYSTEM_PROMPT)] + state["messages"][:-1]
    )

    user_id = config["configurable"]["user_id"]
    todo_namespace = (UpdateType.TODO, user_id)

    updated_todos: list[ToDo] = []
    created_todos: list[ToDo] = []

    for todo in todo_list.todo_list:
        user_todo = store.get(namespace=todo_namespace, key=todo.id.hex)
        if user_todo is None:  # new todo
            user_todo = ToDo.model_validate({**todo.model_dump(exclude_none=True)})
            created_todos.append(user_todo)
        else:  # existing todo
            user_todo = ToDo.model_validate(user_todo.value).model_copy(update={**todo.model_dump(exclude_none=True)})
            updated_todos.append(user_todo)

    for user_todo in updated_todos + created_todos:
        store.put(namespace=todo_namespace, key=user_todo.id.hex, value=user_todo.model_dump(mode="json"))

    updated_todos = "\n".join(f"{user_todo.model_dump(mode='json', exclude_none=True)}" for user_todo in updated_todos)
    created_todos = "\n".join(f"{user_todo.model_dump(mode='json')}" for user_todo in created_todos)

    message = f"Updates for existing ToDos:\n{updated_todos}\n\nCreated ToDos:\n{created_todos}"

    return {"messages": [AIMessage(content=message)]}


# update instructions node

UPDATE_INSTRUCTIONS_SYSTEM_PROMPT = """# ROLE
You extract the user's preferences for HOW their ToDo list should be managed.

# WHAT IS AN INSTRUCTION
An instruction is a lasting rule about managing the ToDo list, for example:
- "Always add a deadline to my tasks."
- "Group grocery items into a single task."
- "Mark tasks as active when I say I'm working on them."
An instruction is NOT a task ("buy milk") and NOT a personal fact ("I live in Cairo").

# INPUT
You receive a conversation between a human and an AI assistant.
Some AI messages are internal logs, for example "Updates for existing Instructions: ..." and
"Created Instructions: ...". These logs show instructions that are already saved, including their `id`.

# FIELDS OF EACH INSTRUCTION
- `id`: the uuid of an existing instruction. Leave empty for a new instruction.
- `instruction`: one clear rule written as a command (for example "Always add a deadline to new tasks.").

# STEPS
Step 1. Read the latest human messages and find every preference about managing the ToDo list.
Step 2. For each preference, check the AI log messages for an instruction on the SAME topic.
        - Same topic found -> copy its `id` EXACTLY and write the new, updated rule.
        - Not found        -> leave `id` empty and write the new rule.
Step 3. Return all new and changed instructions in `instructions`.

# RULES
- One rule per instruction.
- Do NOT return instructions that did not change.
- Do NOT create a duplicate of an instruction that already exists. Update it instead.
- Only use preferences the human actually stated. Never invent rules or ids.
- Never make up a uuid. Either copy an existing one exactly or leave it empty.
- If nothing needs to be saved, return an empty `instructions` list.
"""


def update_instructions(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:

    class Instructions(BaseModel):
        instructions: list[InstructionUpdate] = Field(default_factory=lambda: list[InstructionUpdate]())

    instructions: Instructions = default_llm.with_structured_output(schema=Instructions).invoke(
        input=[SystemMessage(content=UPDATE_INSTRUCTIONS_SYSTEM_PROMPT)] + state["messages"][:-1]
    )

    user_id = config["configurable"]["user_id"]
    instruction_namespace = (UpdateType.INSTRUCTION, user_id)

    updated_instructions: list[Instruction] = []
    created_instructions: list[Instruction] = []

    for instruction in instructions.instructions:
        user_instruction = store.get(namespace=instruction_namespace, key=instruction.id.hex)
        if user_instruction is None:  # new instruction
            user_instruction = Instruction.model_validate({**instruction.model_dump(exclude_none=True)})
            created_instructions.append(user_instruction)
        else:  # existing instruction
            user_instruction = Instruction.model_validate(user_instruction.value).model_copy(
                update={**instruction.model_dump(exclude_none=True)}
            )
            updated_instructions.append(user_instruction)

    for user_instruction in updated_instructions + created_instructions:
        store.put(
            namespace=instruction_namespace, key=user_instruction.id.hex, value=user_instruction.model_dump(mode="json")
        )

    updated_instructions = "\n".join(
        f"{user_instruction.model_dump(mode='json', exclude_none=True)}" for user_instruction in updated_instructions
    )
    created_instructions = "\n".join(
        f"{user_instruction.model_dump(mode='json')}" for user_instruction in created_instructions
    )

    message = (
        f"Updates for existing Instructions:\n{updated_instructions}\n\nCreated Instructions:\n{created_instructions}"
    )

    return {"messages": [AIMessage(content=message)]}


# respond node

RESPOND_SYSTEM_PROMPT = """# ROLE
You are a friendly, helpful companion that helps the user keep track of their ToDo list.
You are now writing the reply that the user will read.

# WHAT YOU REMEMBER ABOUT THE USER
User Profile:
```json
{user_profile}
```

ToDo List:
```json
{user_todos}
```

The user's preferences for managing the ToDo list:
```json
{user_instructions}
```

An empty section means nothing is saved yet.

# ABOUT THE CONVERSATION
Some AI messages in the conversation are internal logs, not replies. They look like:
- JSON such as "update_type": "todo"
- "New profile created ...", "Existing profile updated ..."
- "Updates for existing ToDos ...", "Created ToDos ..."
- "Updates for existing Instructions ...", "Created Instructions ..."
The user never saw these logs. Use them only to know what was just saved.

# HOW TO REPLY
Step 1. Find the most recent message from the human and answer it directly.
Step 2. If the logs show the ToDo list was just changed, tell the user in one short sentence
        what changed (for example "I've added 'Buy milk' to your list for tomorrow.").
Step 3. If the user asks about their tasks, answer using the ToDo List above.
        Use a short bullet list, show deadlines in a friendly format, and skip tasks marked `done`
        unless the user asks for them.
Step 4. Use the user's name and profile details only when it feels natural.

# RULES
- Do NOT mention that you updated the user's profile.
- Do NOT mention that you updated the instructions.
- Never show uuids, JSON, field names or the internal logs to the user.
- Never invent tasks, facts or deadlines that are not in memory or in the conversation.
- Follow the user's preferences in the instructions section.
- Keep replies short, warm and clear: usually 1 to 4 sentences, plus a list if needed.
"""


def respond(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> dict[Literal["messages"], list[AIMessage]]:
    user_id = config["configurable"]["user_id"]
    profile_namespace = (UpdateType.PROFILE, user_id)
    todo_namespace = (UpdateType.TODO, user_id)
    instruction_namespace = (UpdateType.INSTRUCTION, user_id)

    profiles = store.search(profile_namespace, limit=1)
    user_profile = ""
    if profiles:
        user_profile = profiles[0].value

    todos = store.search(todo_namespace)
    user_todos = ""
    if todos:
        user_todos = "\n".join(f"{todo.value}" for todo in todos)

    instructions = store.search(instruction_namespace)
    user_instructions = ""
    if instructions:
        user_instructions = "\n".join(f"{instruction.value}" for instruction in instructions)

    system_message = RESPOND_SYSTEM_PROMPT.format(
        user_profile=user_profile, user_todos=user_todos, user_instructions=user_instructions
    )
    response: AIMessage = default_llm.invoke(input=[SystemMessage(content=system_message)] + state["messages"][:-1])

    return {"messages": [AIMessage(content=response.text)]}


def router(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> Literal["update_profile", "update_todos", "update_instructions", "respond"]:
    update_message: AIMessage = state["messages"][-1]
    update = Update.model_validate(json.loads(update_message.text))

    if update.update_type == UpdateType.PROFILE:
        update_type = "update_profile"
    elif update.update_type == UpdateType.TODO:
        update_type = "update_todos"
    elif update.update_type == UpdateType.INSTRUCTION:
        update_type = "update_instructions"
    elif update.update_type == UpdateType.RESPOND:
        update_type = "respond"

    return update_type


# graph
builder = StateGraph(MessagesState)

builder.add_node(node="maestro", action=maestro)
builder.add_node(node="update_profile", action=update_profile)
builder.add_node(node="update_todos", action=update_todos)
builder.add_node(node="update_instructions", action=update_instructions)
builder.add_node(node="respond", action=respond)

builder.add_edge(start_key=START, end_key="maestro")
builder.add_conditional_edges(source="maestro", path=router)

builder.add_edge(start_key="update_profile", end_key="maestro")
builder.add_edge(start_key="update_todos", end_key="maestro")
builder.add_edge(start_key="update_instructions", end_key="maestro")
builder.add_edge(start_key="respond", end_key=END)


memory_store = InMemoryStore()
memory_saver = InMemorySaver()

graph = builder.compile(checkpointer=memory_saver, store=memory_store, name="memory-agent")

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

    print("-" * 40)
    namespaces = graph.store.list_namespaces()
    print(f"Namespaces: {namespaces}")
    for namespace in namespaces:
        print(graph.store.search(namespace))


if __name__ == "__main__":
    main()
