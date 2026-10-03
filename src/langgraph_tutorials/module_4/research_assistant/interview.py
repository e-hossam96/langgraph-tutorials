import operator
from collections.abc import Sequence
from typing import Annotated, Literal

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from .base import llm, llm_invoke, web_search
from .research_team import Researcher

INTERVIEW_SYSTEM_PROMPT = """You are a researcher interviewing an expert to learn about a research topic.

Your goal is to distill the conversation into insights that are:

**Interesting**: surprising or non-obvious to a general audience.
**Specific**: grounded in concrete examples the expert gives, not generalities.

Here is your persona and focus aspect of the research topic: {persona}

You have a budget of {max_turns} exchanges with the expert; you have used {turns_used} so far.

Procedure:

- On your first turn, introduce yourself using the name from your persona, then ask your first question.
- On every later turn, ask exactly one focused follow-up question that drills deeper into your focus area.
- Stay in character as your persona throughout the conversation.
- Once you are satisfied with your understanding, or you are nearing the end of your turn budget, end the \
conversation with exactly this line, verbatim: "Thank you for your help!"."""


def _reframe_for(messages: Sequence[AnyMessage], speaker_name: str | None) -> list[AnyMessage]:
    return [
        HumanMessage(content=m.content, name=m.name) if isinstance(m, AIMessage) and m.name != speaker_name else m
        for m in messages
    ]


class InterviewGraphState(MessagesState):
    interview_turns: int
    researcher: Researcher
    search_query: str
    documents: Annotated[list[str], operator.add]
    interview: Annotated[list[str], operator.add]
    sections: Annotated[list[str], operator.add]


# question generation node


def generate_research_question(
    state: InterviewGraphState,
) -> dict[Literal["messages", "interview"], list[AIMessage | str]]:
    researcher = state["researcher"]
    messages = _reframe_for(state["messages"], speaker_name="researcher")

    max_turns = state.get("interview_turns", 2)
    turns_used = len([m for m in state["messages"] if isinstance(m, AIMessage) and m.name == "expert"])

    system_message = INTERVIEW_SYSTEM_PROMPT.format(
        persona=researcher.persona, max_turns=max_turns, turns_used=turns_used
    )

    question: AIMessage = llm_invoke(llm, input=[SystemMessage(content=system_message)] + messages)

    question.name = "researcher"

    return {"messages": [question], "interview": [f"Researcher: {question.text}"]}


# search query generation node
class WebSearchQuery(BaseModel):
    query: str


WEB_SEARCH_SYSTEM_PROMPT = """You will be given a transcript of an ongoing interview, formatted as alternating \
"Researcher:" and "Expert:" turns.

Your goal is to produce a single web-search query that will retrieve information the expert needs to answer \
the researcher's most recent question.

Steps:
1. Read the full transcript for context.
2. Focus specifically on the researcher's final (most recent) question.
3. Rewrite that question as a concise, keyword-focused search-engine query — not a restated question or full \
sentence."""


def generate_search_query(state: InterviewGraphState) -> dict[Literal["search_query"], str]:
    structured_output = llm.with_structured_output(schema=WebSearchQuery)

    web_search_query: WebSearchQuery = llm_invoke(
        structured_output,
        input=[
            SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"Conversation: {'\n---\n'.join(state['interview'])}"),
        ],
    )

    return {"search_query": web_search_query.query}


# web search node
def web_search_research(state: InterviewGraphState) -> dict[Literal["documents"], list[str]]:
    results = web_search(search_query=state["search_query"])
    return {"documents": [r.result for r in results]}


# answer generation node
ANSWER_GENERATION_SYSTEM_PROMPT = """You are the expert being interviewed by a researcher.

Here is the researcher's area of focus and persona: {persona}

Your goal is to answer the researcher's most recent question using only the context below.

Context:

\"\"\"
{documents}
\"\"\"

When answering, follow these guidelines:

1. Base your answer strictly on the information in the context above.
2. Do not introduce external information or assumptions beyond what the context states.
3. If the context does not contain enough information to answer, say so honestly instead of guessing.
4. Each context document starts with its own source title and URL. Cite the relevant source inline next to \
any statement it supports, e.g. [1].
5. End your answer with a numbered source list matching the citations you used, formatted as:
   [1] [Source Title](Source URL)
   [2] [Source Title](Source URL)
6. Keep your answer conversational and focused, as in a real interview — do not pad it with unnecessary detail."""


def generate_answer(state: InterviewGraphState) -> dict[Literal["messages", "interview"], list[AIMessage | str]]:
    researcher = state["researcher"]
    messages = _reframe_for(state["messages"], speaker_name="expert")
    context = state["documents"]

    system_message = ANSWER_GENERATION_SYSTEM_PROMPT.format(persona=researcher.persona, documents=context)
    answer: AIMessage = llm_invoke(llm, [SystemMessage(content=system_message)] + messages)

    answer.name = "expert"

    return {"messages": [answer], "interview": [f"Expert: {answer.text}"]}


# section writer node
SECTION_WRITER_SYSTEM_PROMPT = """You are an expert technical writer.

Your task is to turn an interview transcript between a researcher and an expert into a short, easily \
digestible section of a report.

1. Read the interview transcript below. The expert's answers already contain inline citations (e.g. [1]) \
and a trailing numbered source list — these are the sources for this section.

2. Write the section using this markdown structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

3. Title:
- Make it engaging, based on the researcher's focus area: {focus}

4. Summary:
- Open with brief background/context for the focus area.
- Emphasize what is novel, interesting, or surprising about the insights gathered from the interview.
- Do not mention the names of the researcher or the expert.
- Aim for approximately 400 words maximum.
- Use numbered citations (e.g. [1], [2]) tied to the Sources list below.

5. Sources:
- List every source cited in the summary, and only those sources.
- Provide the full link (or document path) for each source.
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown:

### Sources
[1] Link or document name
[2] Link or document name

- Merge duplicate sources into a single entry — never list the same source under two numbers. For example, \
this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/

6. Before finishing, confirm:
- The structure above was followed exactly, with no preamble before the title.
- Every citation used in the summary has a matching entry in Sources, and vice versa."""


def write_section(state: InterviewGraphState) -> dict[Literal["sections"], list[str]]:
    interview = state["interview"]
    researcher = state["researcher"]

    system_message = SECTION_WRITER_SYSTEM_PROMPT.format(focus=researcher.focus)
    section: AIMessage = llm_invoke(
        llm,
        [SystemMessage(content=system_message)] + [HumanMessage(content=f"Conversation: {'\n---\n'.join(interview)}")],
    )

    return {"sections": [section.text]}


# conditional edge
def route_messages(state: InterviewGraphState) -> Literal["write_section", "generate_research_question"]:
    messages = state["messages"]
    interview_turns = state.get("interview_turns", 2)

    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == "expert"])

    if num_responses >= interview_turns:
        return "write_section"

    last_question = messages[-2]

    if "thank you for your help!" in last_question.text.lower():
        return "write_section"
    return "generate_research_question"


# graph configs

interview_graph_builder = StateGraph(InterviewGraphState)
interview_graph_builder.add_node(node="generate_research_question", action=generate_research_question)
interview_graph_builder.add_node(node="generate_search_query", action=generate_search_query)
interview_graph_builder.add_node(node="web_search_research", action=web_search_research)
interview_graph_builder.add_node(node="generate_answer", action=generate_answer)
interview_graph_builder.add_node(node="write_section", action=write_section)


interview_graph_builder.add_edge(START, "generate_research_question")
interview_graph_builder.add_edge("generate_research_question", "generate_search_query")
interview_graph_builder.add_edge("generate_search_query", "web_search_research")
interview_graph_builder.add_edge("web_search_research", "generate_answer")
interview_graph_builder.add_edge("write_section", END)

interview_graph_builder.add_conditional_edges(
    source="generate_answer", path=route_messages, path_map=["generate_research_question", "write_section"]
)
