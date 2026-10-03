from dataclasses import dataclass

from langgraph_tutorials.llms import default_llm as llm
from langgraph_tutorials.llms import llm_invoke
from langgraph_tutorials.search import web_search_client
from langgraph_tutorials.tracers import langfuse

__all__ = ["WebSearchResult", "langfuse", "llm", "llm_invoke", "web_search"]


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str

    @property
    def result(self) -> str:
        return f'Title: {self.title}\nURL: {self.url}\nContent: \n"""{self.content}"""\n'


def web_search(search_query: str) -> list[WebSearchResult]:
    search_results = web_search_client.search(query=search_query, max_results=2)
    results = [
        WebSearchResult(title=result["title"], url=result["url"], content=result["content"])
        for result in search_results["results"]
    ]
    return results
