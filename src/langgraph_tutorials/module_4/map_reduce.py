import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.search import web_search_client
from langgraph_tutorials.tracers import langfuse_handler


@dataclass
class WebSearchResult:
    answer: str | None = None


def web_search(search_query: str) -> WebSearchResult:
    search_results = web_search_client.search(query=search_query, max_results=2, include_answer=True)
    return WebSearchResult(answer=search_results["answer"])


# web search node


@dataclass
class WebSearchState:
    web_search_query: str


def web_search_node(state: WebSearchState) -> dict[Literal["web_search_answers"], str]:
    web_search_answer = web_search(search_query=state.web_search_query)
    return {"web_search_answers": [web_search_answer.answer]}


# agent


@dataclass
class CompleteWebSearchState:
    question: str
    web_search_queries: list[str] | None = None
    web_search_answers: Annotated[list[str], operator.add] = field(default_factory=list)
    best_answer: str | None = None


def generate_search_queries(
    state: CompleteWebSearchState,
) -> CompleteWebSearchState:

    class SearchQueries(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        web_search_queries: list[str]

    system_message = SystemMessage(
        content="You are a helpful assistant that generates search queries (5 at max) based on user questions."
    )
    user_message = HumanMessage(content=f"Generate search queries for the following question: {state.question}")

    response = llm.with_structured_output(SearchQueries).invoke(input=[system_message, user_message])
    state.web_search_queries = response.web_search_queries
    return state


# create search node for each search query
def continue_search_node(state: CompleteWebSearchState) -> list[Send]:
    return [Send("web_search_node", WebSearchState(web_search_query=query)) for query in state.web_search_queries]


def select_best_answer(state: CompleteWebSearchState) -> CompleteWebSearchState:
    class BestAnswer(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        best_answer_id: int

    system_message = SystemMessage(
        content="""You are a helpful assistant that selects the best answer \
based on user question from a list of web search answers.

You will be given a question and a list of web search answers. \
You should select the best answer by returning the index of the best answer in the list of web search answers. \
If there are multiple answers that are equally good, return the index of the first one."""
    )

    answers = "\n".join([f"{i}: {answer}" for i, answer in enumerate(state.web_search_answers)])

    user_message = HumanMessage(
        content=(
            f"Select the best answer for the following question: {state.question} given "
            f'the following search results:\n"""{answers}"""\nReturn the index of the best answer.'
        )
    )

    response = llm.with_structured_output(BestAnswer).invoke(input=[system_message, user_message])
    state.best_answer = state.web_search_answers[response.best_answer_id]
    return state


main_agent_builder = StateGraph(CompleteWebSearchState)

main_agent_builder.add_node("generate_search_queries", generate_search_queries)
main_agent_builder.add_node("web_search_node", web_search_node)
main_agent_builder.add_node("select_best_answer", select_best_answer)

main_agent_builder.add_edge(START, "generate_search_queries")
main_agent_builder.add_conditional_edges(
    source="generate_search_queries", path=continue_search_node, path_map=["web_search_node"]
)
main_agent_builder.add_edge("web_search_node", "select_best_answer")
main_agent_builder.add_edge("select_best_answer", END)


async def main():
    graph = main_agent_builder.compile(checkpointer=InMemorySaver())
    with open("graph.md", "w+", encoding="utf-8") as f:
        f.write(f"""```mermaid
{graph.get_graph(xray=True).draw_mermaid(with_styles=False)}
```
""")

    thread_id = "1"
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
        },
    }

    async for event in graph.astream(
        input={"question": "How were Nvidia's Q2 2025 earnings"}, config=config, stream_mode="updates"
    ):
        print(event)
        print("-" * 40)

    print("Answer:", graph.get_state(config=config).values["best_answer"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
