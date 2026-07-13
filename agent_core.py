"""
agent_core.py
-------------
The agent — a LangGraph pipeline with three steps:

  1. plan_queries        -> turn a startup name into targeted research queries
                             (founders, business model/category, funding, traction)
  2. gather_research      -> run those queries against Tavily
  3. synthesize_assessment -> ask an LLM to produce a structured founder/startup
                              profile, and (if an investor thesis is given)
                              an explicit fit verdict against it

This mirrors real ICP (Ideal Customer Profile) research work: classify a
startup by category and maturity, profile its founders, then judge fit
against a specific investor's criteria — rather than just summarizing news.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from search_tools import get_tavily_client, run_all_searches


# ---------------------------------------------------------------------------
# State definition — this is what gets passed between nodes.
# ---------------------------------------------------------------------------
class BriefingState(TypedDict):
    startup_name: str
    startup_description: str   # optional context the user already knows
    investor_thesis: str       # optional — what to check fit against
    queries: list[str]
    search_results: dict
    assessment: str


# ---------------------------------------------------------------------------
# Node 1: plan_queries
# Queries are now aimed at founder background + business model classification
# + traction signals, rather than generic "news about X".
# ---------------------------------------------------------------------------
def plan_queries(state: BriefingState) -> BriefingState:
    name = state["startup_name"]
    queries = [
        (f'"{name}" founder co-founder CEO', "general"),
        (f'"{name}" crunchbase OR linkedin company profile', "general"),
        (f'"{name}" what does it do product overview', "general"),
        (f'"{name}" funding round raised seed OR series', "news"),
        (f'"{name}" competitors alternative', "general"),
    ]
    return {**state, "queries": queries}


# ---------------------------------------------------------------------------
# Node 2: gather_research (unchanged logic, still concurrent)
# ---------------------------------------------------------------------------
def make_gather_research(tavily_api_key: str):
    client = get_tavily_client(tavily_api_key)

    def gather_research(state: BriefingState) -> BriefingState:
        results = run_all_searches(state["queries"], client)
        return {**state, "search_results": results}

    return gather_research


# ---------------------------------------------------------------------------
# Node 3: synthesize_assessment
# Two prompt variants: with and without an investor thesis to check fit against.
# ---------------------------------------------------------------------------
BASE_SECTIONS = """## Founder Profile
Who founded this company — background, prior experience, relevant credibility \
signals. If founder info isn't in the research, say so explicitly rather than \
guessing.

## Business Classification
Categorize the startup: product category (e.g. AI/SaaS/B2B/B2C/D2C/Marketplace/ \
Hardware), target customer, and business model (subscription, transaction fee, \
etc.) — based only on what the research shows.

## GTM Maturity
Signals of stage and go-to-market readiness: funding stage, geographic reach, \
customer traction, any expansion signals (e.g. India-US, multi-market).

## Sources
Numbered list of URLs actually used, in the order referenced.
"""

FIT_SECTION = """
## Investor Fit Assessment
The user is evaluating this startup against this investor thesis / ICP criteria:
\"\"\"{investor_thesis}\"\"\"

Give a clear verdict: Strong Fit / Partial Fit / Not a Fit, followed by 2-4 \
bullet points of reasoning that explicitly reference the thesis criteria \
against what was found in the research. If the research doesn't have enough \
information to judge a criterion, say so rather than assuming.
"""

SYNTHESIS_PROMPT = """You are conducting founder and startup due diligence, the \
kind done before deciding whether a startup fits an investor's or event's \
criteria. You have been given raw web search results. Turn this into a \
structured assessment.

Startup: {startup_name}
{description_line}

Raw research (grouped by the query that produced it):
{research_dump}

Write the assessment in this exact structure, using Markdown:

{base_sections}
{fit_section}

Rules:
- Every claim must be traceable to the research provided. Do not invent founder \
names, funding figures, or facts not present in the research.
- If research is thin on a section, say so explicitly rather than filling the gap.
- Keep it concise — this is a screening assessment, not an essay.
"""


def make_synthesize_assessment(groq_api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
    )

    def synthesize_assessment(state: BriefingState) -> BriefingState:
        chunks = []
        for query, results in state["search_results"].items():
            chunks.append(f"### Results for: {query}")
            for r in results:
                chunks.append(f"- {r['title']} ({r['url']})\n  {r['content'][:600]}")
        research_dump = "\n".join(chunks)

        description_line = (
            f"User-provided context: {state['startup_description']}"
            if state.get("startup_description")
            else ""
        )

        fit_section = ""
        if state.get("investor_thesis"):
            fit_section = FIT_SECTION.format(investor_thesis=state["investor_thesis"])

        prompt = SYNTHESIS_PROMPT.format(
            startup_name=state["startup_name"],
            description_line=description_line,
            research_dump=research_dump,
            base_sections=BASE_SECTIONS,
            fit_section=fit_section,
        )
        response = llm.invoke(prompt)
        return {**state, "assessment": response.content}

    return synthesize_assessment


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_agent(groq_api_key: str, tavily_api_key: str):
    graph = StateGraph(BriefingState)

    graph.add_node("plan_queries", plan_queries)
    graph.add_node("gather_research", make_gather_research(tavily_api_key))
    graph.add_node("synthesize_assessment", make_synthesize_assessment(groq_api_key))

    graph.set_entry_point("plan_queries")
    graph.add_edge("plan_queries", "gather_research")
    graph.add_edge("gather_research", "synthesize_assessment")
    graph.add_edge("synthesize_assessment", END)

    return graph.compile()


def run_assessment(
    agent_app,
    startup_name: str,
    startup_description: str = "",
    investor_thesis: str = "",
) -> BriefingState:
    """Convenience wrapper: run the whole pipeline for one startup."""
    initial_state: BriefingState = {
        "startup_name": startup_name,
        "startup_description": startup_description,
        "investor_thesis": investor_thesis,
        "queries": [],
        "search_results": {},
        "assessment": "",
    }
    return agent_app.invoke(initial_state)

