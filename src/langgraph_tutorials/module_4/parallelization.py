import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.search import web_search_client
from langgraph_tutorials.tracers import langfuse_handler


class WebSearchTopics(StrEnum):
    GENERAL = "general"
    FINANCE = "finance"


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str


@dataclass
class ParallelizationState:
    question: str
    search_query: str | None = None
    search_results: Annotated[list[WebSearchResult], operator.add] = field(default_factory=list)
    answer: str | None = None


def generate_search_query(state: ParallelizationState) -> dict[Literal["search_query"], str]:

    class SearchQueryResponseModel(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        search_query: str

    system_message = SystemMessage(
        content="You are a helpful assistant that generates search queries based on user questions."
    )
    user_message = HumanMessage(content=f"Generate a search query for the following question: {state.question}")

    response = llm.with_structured_output(SearchQueryResponseModel).invoke(input=[system_message, user_message])
    return {"search_query": response.search_query}


def web_search(search_query: str, search_topic: WebSearchTopics) -> list[WebSearchResult]:
    search_results = web_search_client.search(query=search_query, max_results=5, topic=search_topic)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    return results


def general_web_search(state: ParallelizationState) -> dict[Literal["search_results"], list[WebSearchResult]]:
    search_results = web_search(search_query=state.search_query, search_topic=WebSearchTopics.GENERAL)
    return {"search_results": search_results}


def finance_web_search(state: ParallelizationState) -> dict[Literal["search_results"], list[WebSearchResult]]:
    search_results = web_search(search_query=state.search_query, search_topic=WebSearchTopics.FINANCE)
    return {"search_results": search_results}


def generate_answer(state: ParallelizationState) -> dict[Literal["answer"], str]:
    class AnswerResponseModel(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        answer: str

    system_message = SystemMessage(
        content="You are a helpful assistant that generates answers based on user questions and web search results."
    )
    user_message = HumanMessage(
        content=f"Generate an answer for the following question: {state.question} given the following search results: {state.search_results}"
    )

    response = llm.with_structured_output(AnswerResponseModel).invoke(input=[system_message, user_message])
    return {"answer": response.answer}


async def main():
    builder = StateGraph(ParallelizationState)

    builder.add_node("search_query", generate_search_query)
    builder.add_node("general_web_search", general_web_search)
    builder.add_node("finance_web_search", finance_web_search)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "search_query")
    builder.add_edge("search_query", "general_web_search")
    builder.add_edge("search_query", "finance_web_search")
    builder.add_edge("general_web_search", "generate_answer")
    builder.add_edge("finance_web_search", "generate_answer")

    graph = builder.compile(checkpointer=InMemorySaver())
    with open("graph.md", "w+", encoding="utf-8") as f:
        f.write(f"""```mermaid
{graph.get_graph(xray=True).draw_mermaid()}
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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
