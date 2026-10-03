from typing import Literal, TypedDict

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .base import llm, llm_invoke


# structured llm configs
class Researcher(BaseModel):
    name: str
    role: str
    focus: str

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nFocus: {self.focus}"


class ResearchTeam(BaseModel):
    researchers: list[Researcher]


SYSTEM_PROMPT = """You are assembling a team of AI researcher personas to investigate a research topic.

Research topic: {topic}

Follow these steps:
1. Identify the most interesting and distinct angles or sub-topics within the topic above.
2. Select the top {num_researchers} of these angles.
3. Assign one researcher persona to each selected angle.

Guidelines for each persona:
- name: a plausible, fictitious full name.
- role: an organizational or professional role suited to investigating that angle (e.g. "CFO", "Data Privacy Lawyer").
- focus: one specific, narrow research angle for this persona to pursue.

Every researcher's focus must be distinct — do not let two personas cover overlapping ground."""


# graph configs
class ResearchTeamGraphState(TypedDict):
    topic: str
    num_researchers: int
    researchers: list[Researcher]


def create_research_team(state: ResearchTeamGraphState) -> dict[Literal["researchers"], list[Researcher]]:
    structured_output = llm.with_structured_output(schema=ResearchTeam)
    system_message = SYSTEM_PROMPT.format(
        topic=state["topic"],
        num_researchers=state["num_researchers"],
    )
    research_team: ResearchTeam = llm_invoke(
        structured_output,
        input=[SystemMessage(content=system_message), HumanMessage(content="Generate the set of researchers.")],
    )
    return {"researchers": research_team.researchers}


research_team_graph_builder = StateGraph(ResearchTeamGraphState)

research_team_graph_builder.add_node(node="create_research_team", action=create_research_team)

research_team_graph_builder.add_edge(start_key=START, end_key="create_research_team")
research_team_graph_builder.add_edge(start_key="create_research_team", end_key=END)
