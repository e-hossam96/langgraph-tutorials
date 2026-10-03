import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .base import llm, llm_invoke
from .interview import interview_graph_builder
from .research_team import Researcher, research_team_graph_builder


class ReportGraphState(TypedDict):
    topic: str
    num_researchers: int
    num_interview_turns: int
    researchers: list[Researcher]
    sections: Annotated[list[str], operator.add]
    introduction: str
    body: str
    conclusion: str
    report: str


# report body writer node
REPORT_BODY_WRITER_SYSTEM_PROMPT = """You are a technical writer creating a report body on this overall topic:

{topic}

You have a team of researchers. Each researcher has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They wrote up their findings into a memo.

Each memo was written independently, so their bracketed citation numbers (e.g. [1], [2]) are NOT consistent \
with one another — the same number in two memos can point to two different sources, and the same source can \
appear under different numbers across memos.

Your task:

1. You will be given a collection of memos from your researchers.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos \
— do not summarize memo-by-memo.
4. As you write, assign each distinct source a fresh citation number the first time you cite it, and reuse \
that same number every time you cite that source again, regardless of what number it had in its original memo.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Do not add sub-headings of your own — write a single flowing narrative under the title header.
4. Start your report with a single title header: ## Insights
5. Do not mention any researcher names in your report.
6. Use your renumbered citations inline, e.g. [1], [2].
7. End with a final, deduplicated list of sources under a `## Sources` header, numbered to match your inline \
citations, with no repeated entries:

## Sources
[1] Source 1
[2] Source 2"""


def write_report_body(state: ReportGraphState) -> dict[Literal["body"], str]:
    topic = state["topic"]
    sections = "\n---\n".join(state["sections"])
    system_message = REPORT_BODY_WRITER_SYSTEM_PROMPT.format(topic=topic)
    report_body: AIMessage = llm_invoke(
        llm,
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Write a report based upon these memos:\n{sections}"),
        ],
    )
    return {"body": report_body.text}


# report introduction-conclusion writer nodes
INTRODUCTION_CONCLUSION_WRITER_SYSTEM_PROMPT = """You are a technical writer finishing a report on {topic}.

You will be given the research memos the team produced — the same memos used to write the report body.

Your job is to write ONE of two things, as instructed in the user message: a crisp, compelling introduction, \
or a conclusion.

Guidelines:

1. Use markdown formatting.
2. Include no pre-amble before the section itself.
3. Target around 100 words.
4. Do not include citations, bracket references (e.g. [1]), or a Sources section — the memos are numbered \
independently of one another and of the final report, so citations here would be inconsistent; sources \
belong only in the report body.

If writing the introduction:
- Start with a single # header containing a compelling title for the whole report.
- Follow it with a ## Introduction header, then a short paragraph crisply previewing the themes across all memos.

If writing the conclusion:
- Use a ## Conclusion header, then a short paragraph recapping the themes across all memos."""


def _write_introduction_or_conclusion(
    sections: list[str], topic: str, type: Literal["introduction", "conclusion"]
) -> AIMessage:
    joined_sections = "\n---\n".join(sections)
    system_message = INTRODUCTION_CONCLUSION_WRITER_SYSTEM_PROMPT.format(topic=topic)
    user_message = f"Write a {type} based on the following sections:\n{joined_sections}"
    report_introduction_or_conclusion: AIMessage = llm_invoke(
        llm,
        [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ],
    )
    return report_introduction_or_conclusion


def write_introduction(state: ReportGraphState) -> dict[Literal["introduction"], str]:
    introduction: AIMessage = _write_introduction_or_conclusion(state["sections"], state["topic"], "introduction")
    return {"introduction": introduction.text}


def write_conclusion(state: ReportGraphState) -> dict[Literal["conclusion"], str]:
    conclusion: AIMessage = _write_introduction_or_conclusion(state["sections"], state["topic"], "conclusion")
    return {"conclusion": conclusion.text}


# finalize report node
_INSIGHTS_HEADER_RE = re.compile(r"^\s*##\s*Insights\s*\n+", flags=re.IGNORECASE)
_SOURCES_HEADER_RE = re.compile(r"\n+##\s*Sources\s*\n+", flags=re.IGNORECASE)


def finalize_report(state: ReportGraphState) -> dict[Literal["report"], str]:
    body = _INSIGHTS_HEADER_RE.sub("", state["body"], count=1)

    sources = None
    match = _SOURCES_HEADER_RE.search(body)
    if match is not None:
        body, sources = body[: match.start()], body[match.end() :]

    report = state["introduction"] + "\n\n---\n\n" + body.strip() + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        report += "\n\n## Sources\n" + sources.strip()
    return {"report": report}


# graph configs
def start_interviews(state: ReportGraphState) -> list[Send]:
    topic = state["topic"]
    return [
        Send(
            "interview",
            {
                "researcher": researcher,
                "interview_turns": state["num_interview_turns"],
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")],
            },
        )
        for researcher in state["researchers"]
    ]


report_graph_builder = StateGraph(ReportGraphState)
report_graph_builder.add_node("research_team", research_team_graph_builder.compile())
report_graph_builder.add_node("interview", interview_graph_builder.compile())
report_graph_builder.add_node("body", write_report_body)
report_graph_builder.add_node("introduction", write_introduction)
report_graph_builder.add_node("conclusion", write_conclusion)
report_graph_builder.add_node("report", finalize_report)

# Logic
report_graph_builder.add_edge(START, "research_team")

report_graph_builder.add_conditional_edges("research_team", start_interviews, ["interview"])

report_graph_builder.add_edge("interview", "body")
report_graph_builder.add_edge("interview", "introduction")
report_graph_builder.add_edge("interview", "conclusion")
report_graph_builder.add_edge(["introduction", "body", "conclusion"], "report")
report_graph_builder.add_edge("report", END)
