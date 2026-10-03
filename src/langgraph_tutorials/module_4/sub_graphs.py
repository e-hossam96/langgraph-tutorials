from dataclasses import dataclass, field
from enum import StrEnum

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.search import web_search_client
from langgraph_tutorials.tracers import langfuse_handler


class WebSearchTopics(StrEnum):
    NEWS = "news"
    FINANCE = "finance"


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str


def web_search(search_query: str, search_topic: WebSearchTopics) -> list[WebSearchResult]:
    search_results = web_search_client.search(query=search_query, max_results=2, topic=search_topic)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    return results


# finance web search agent


@dataclass
class FinanceWebSearchState:
    finance_search_query: str
    finance_search_results: list[WebSearchResult]


def finance_web_search(state: FinanceWebSearchState) -> FinanceWebSearchState:
    state.finance_search_results = web_search(
        search_query=state.finance_search_query, search_topic=WebSearchTopics.FINANCE
    )
    return state


finance_web_search_agent_builder = StateGraph(FinanceWebSearchState)
finance_web_search_agent_builder.add_node("finance_web_search", finance_web_search)

finance_web_search_agent_builder.add_edge(START, "finance_web_search")
finance_web_search_agent_builder.add_edge("finance_web_search", END)

# news web search agent


@dataclass
class NewsWebSearchState:
    news_search_query: str
    news_search_results: list[WebSearchResult]


def news_web_search(state: NewsWebSearchState) -> NewsWebSearchState:
    state.news_search_results = web_search(search_query=state.news_search_query, search_topic=WebSearchTopics.NEWS)
    return state


news_web_search_agent_builder = StateGraph(NewsWebSearchState)
news_web_search_agent_builder.add_node("news_web_search", news_web_search)

news_web_search_agent_builder.add_edge(START, "news_web_search")
news_web_search_agent_builder.add_edge("news_web_search", END)

# main agent


@dataclass
class WebSearchState:
    question: str
    finance_search_query: str | None = None
    news_search_query: str | None = None
    finance_search_results: list[WebSearchResult] = field(default_factory=list)
    news_search_results: list[WebSearchResult] = field(default_factory=list)
    answer: str | None = None


def generate_search_queries(
    state: WebSearchState,
) -> WebSearchState:

    class SearchQueryResponseModel(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        finance_search_query: str
        news_search_query: str

    system_message = SystemMessage(
        content="You are a helpful assistant that generates search queries based on user questions."
    )
    user_message = HumanMessage(
        content=f"Generate finance and news search queries for the following question: {state.question}"
    )

    response = llm.with_structured_output(SearchQueryResponseModel).invoke(input=[system_message, user_message])
    state.finance_search_query = response.finance_search_query
    state.news_search_query = response.news_search_query
    return state


def generate_answer(state: WebSearchState) -> WebSearchState:
    class AnswerResponseModel(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore")
        answer: str

    system_message = SystemMessage(
        content="You are a helpful assistant that generates answers based on user questions and web search results."
    )
    user_message = HumanMessage(
        content=f"Generate an answer for the following question: {state.question} given the following search results: {state.finance_search_results + state.news_search_results}"
    )

    response = llm.with_structured_output(AnswerResponseModel).invoke(input=[system_message, user_message])
    state.answer = response.answer
    return state


main_agent_builder = StateGraph(WebSearchState)
main_agent_builder.add_node("generate_search_queries", generate_search_queries)
main_agent_builder.add_node("finance_web_search", finance_web_search_agent_builder.compile())
main_agent_builder.add_node("news_web_search", news_web_search_agent_builder.compile())
main_agent_builder.add_node("generate_answer", generate_answer)

main_agent_builder.add_edge(START, "generate_search_queries")
main_agent_builder.add_edge("generate_search_queries", "finance_web_search")
main_agent_builder.add_edge("generate_search_queries", "news_web_search")
main_agent_builder.add_edge("finance_web_search", "generate_answer")
main_agent_builder.add_edge("news_web_search", "generate_answer")
main_agent_builder.add_edge("generate_answer", END)


async def main():
    graph = main_agent_builder.compile(checkpointer=InMemorySaver())
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

    print("Answer:", graph.get_state(config=config).values["answer"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
