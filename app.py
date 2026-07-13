"""
app.py
------
Streamlit front-end for the Founderlens.

Flow:
  1. User enters a startup name (+ optional description they already know,
     + optional investor thesis/ICP criteria to check fit against).
  2. We run the LangGraph pipeline (plan -> search -> synthesize).
  3. We show a structured assessment: founder profile, business
     classification, GTM maturity, and (if a thesis was given) an explicit
     fit verdict.
  4. Every assessment is saved to history.json on disk, so past searches
     are still there next time you open the app (not just this session).

Run locally with:
    streamlit run app.py

Requires two secrets (see .streamlit/secrets.toml.example):
    GROQ_API_KEY
    TAVILY_API_KEY
"""

import streamlit as st

from agent_core import build_agent, run_assessment
from history_store import load_history, save_entry, clear_history

st.set_page_config(page_title="Founderlens", layout="centered")


def get_agent():
    if "agent_app" not in st.session_state:
        try:
            st.session_state.agent_app = build_agent(
                groq_api_key=st.secrets["GROQ_API_KEY"],
                tavily_api_key=st.secrets["TAVILY_API_KEY"],
            )
        except KeyError as e:
            st.error(
                f"Missing secret: {e}. Add GROQ_API_KEY and TAVILY_API_KEY "
                "to .streamlit/secrets.toml (see the .example file)."
            )
            st.stop()
    return st.session_state.agent_app


# ---------------------------------------------------------------------------
# Sidebar: persistent history loaded from disk (history.json), not just
# this session's memory. Clicking a past entry shows it again below.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Past screenings")
    history = load_history()

    if not history:
        st.caption("Nothing saved yet.")

    selected_past = None
    for entry in reversed(history):
        label = f"{entry['timestamp']} — {entry['name']}"
        if st.button(label, use_container_width=True, key=f"hist_{entry['timestamp']}_{entry['name']}"):
            selected_past = entry

    st.divider()
    if st.button("Clear all history", use_container_width=True):
        clear_history()
        st.rerun()


st.title("Founderlens")
st.write(
    "Enter a startup name to get a structured profile, founder background, "
    "business category (AI/SaaS/B2B/etc.), and GTM maturity, synthesized "
    "from live web search. Optionally add your investor thesis or ICP "
    "criteria to get an explicit fit verdict."
)

startup_name = st.text_input("Startup name", placeholder="e.g. Perplexity, or a founder's company")

with st.expander("Optional: add context or a fit thesis"):
    startup_description = st.text_area(
        "Anything you already know about the startup (optional)",
        placeholder="e.g. B2B SaaS tool for restaurant inventory management, seed stage",
    )
    investor_thesis = st.text_area(
        "Investor thesis / ICP criteria to check fit against (optional)",
        placeholder="e.g. We invest in AI-native B2B SaaS with early revenue and India-US expansion potential, pre-seed to seed stage.",
    )

generate = st.button("Generate assessment", type="primary")

# ---------------------------------------------------------------------------
# Case 1: user clicked a past entry in the sidebar -> just show it, no
# new search/API calls needed.
# ---------------------------------------------------------------------------
if selected_past:
    st.info(f"Showing saved result for **{selected_past['name']}** from {selected_past['timestamp']}")
    st.markdown(selected_past["assessment"])
    st.download_button(
        label="Download this assessment (.md)",
        data=selected_past["assessment"],
        file_name=f"{selected_past['name'].replace(' ', '_')}_assessment.md",
        mime="text/markdown",
    )

# ---------------------------------------------------------------------------
# Case 2: user ran a new search -> run the pipeline, show it, save it.
# ---------------------------------------------------------------------------
elif generate and startup_name.strip():
    agent_app = get_agent()
    progress = st.empty()
    try:
        progress.info("Researching founders, business model, and traction...")
        result = run_assessment(
            agent_app,
            startup_name.strip(),
            startup_description.strip(),
            investor_thesis.strip(),
        )
        progress.empty()
        assessment_text = result["assessment"]

        save_entry(startup_name.strip(), assessment_text)

        st.markdown(assessment_text)

        st.download_button(
            label="Download assessment (.md)",
            data=assessment_text,
            file_name=f"{startup_name.strip().replace(' ', '_')}_assessment.md",
            mime="text/markdown",
        )

    except Exception as e:
        progress.empty()
        st.error(f"Something went wrong generating this assessment: {e}")

elif generate:
    st.warning("Enter a startup name first.")
