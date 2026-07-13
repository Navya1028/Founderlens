# FounderLens

**🔗 Live app: https://founderlens4.streamlit.app/**

An agentic tool that researches a startup , founders, business model, GTM
maturity and, if given an investor thesis or ICP criteria, produces an
explicit fit verdict against it.

Built to mirror real ICP (Ideal Customer Profile) and founder-profiling
research work: segmenting founders by product category (AI/SaaS/B2B),
GTM maturity, and expansion readiness, then judging fit against specific
investor/event criteria, not just summarizing generic company news.

## How it works

1. **Plan queries**: the startup name is turned into 5 targeted search
   queries (founder background, product/business model, funding, traction,
   competitors).
2. **Gather research**: each query runs against the [Tavily](https://tavily.com)
   search API (built for LLM/agent use, returns clean content instead of
   raw HTML), all fired concurrently for speed.
3. **Synthesize assessment**: an LLM (Groq, Llama 3.3 70B) reads the
   research and produces a structured Markdown assessment: founder profile,
   business classification, GTM maturity signals, and if an investor
   thesis was provided, an explicit Strong Fit / Partial Fit / Not a Fit
   verdict with reasoning tied to the thesis criteria.

This is a [LangGraph](https://langchain-ai.github.io/langgraph/) pipeline,
three nodes run in sequence:
`plan_queries -> gather_research -> synthesize_assessment`.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get two free API keys:
   - **Groq**: https://console.groq.com/keys
   - **Tavily**: https://tavily.com (free tier: 1000 searches/month)

3. Copy the secrets template and fill in your keys:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml with your actual keys
   ```

4. Run it:
   ```bash
   streamlit run app.py
   ```

## Project structure

```
├── app.py              # Streamlit UI
├── agent_core.py        # LangGraph pipeline (the actual "agent")
├── search_tools.py       # Tavily search wrapper (concurrent search)
├── list_models.py        # utility: lists models available to an API key
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example
```

## Design notes / things worth knowing if asked about this

- **Why LangGraph and not one big prompt?** Separating research from
  synthesis means each step can be debugged or swapped independently, and
  it mirrors how a real analyst works — research first, then write.
- **Why fixed query templates instead of letting the LLM invent queries?**
  Reliability and cost — deterministic queries avoid the LLM generating a
  vague or off-topic search, and skip an extra LLM call just to plan searches.
- **Concurrent search**: the 5 research queries run in parallel via a thread
  pool rather than one after another, cutting wait time roughly to the
  slowest single query instead of the sum of all five.
- **Hallucination guardrail**: the synthesis prompt explicitly instructs the
  model to only state what's in the retrieved research, name founders/figures
  only if they actually appear in results, and flag thin evidence rather than
  fill gaps — the main real risk with LLM research tools.
- **Optional investor thesis**: the fit-verdict section only appears if the
  user provides criteria, so the tool works both as a general founder/startup
  profiler and as a targeted fit screener.
