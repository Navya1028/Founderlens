"""
search_tools.py
---------------
Thin wrapper around the Tavily search API. Tavily is built specifically for
LLM/agent use cases (it returns clean, pre-summarized content instead of raw
HTML), which makes it a good fit here vs. scraping pages ourselves.

Get a free API key at https://tavily.com (free tier is generous enough for
a student project — 1000 searches/month at time of writing).
"""

import os
from tavily import TavilyClient


def get_tavily_client(api_key: str | None = None) -> TavilyClient:
    """
    Build a Tavily client. Accepts an explicit key (so app.py can pass in
    st.secrets) or falls back to the TAVILY_API_KEY environment variable.
    """
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ValueError(
            "No Tavily API key found. Pass one in, or set TAVILY_API_KEY."
        )
    return TavilyClient(api_key=key)


def search_web(query: str, client: TavilyClient, max_results: int = 6, topic: str = "general") -> list[dict]:
    """
    Run a single search query and return a clean list of results.

    topic="general" (the default) covers company pages, Crunchbase-style
    profiles, LinkedIn, etc. topic="news" restricts results to recent news
    articles only — useful for "latest funding news" style queries, but it
    was WRONGLY set as the default before, which meant founder-bio and
    company-overview queries got filtered down to news-only results and
    often came back empty or irrelevant for smaller/lesser-known startups.
    """
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",   # better relevance for research-style queries
        include_answer=False,      # we do our own synthesis, don't need Tavily's
        topic=topic,
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results


def run_all_searches(queries: list, client: TavilyClient) -> dict:
    """
    Run a batch of queries CONCURRENTLY and return {query: [results]}.

    `queries` can be a list of plain strings (topic defaults to "general"
    for all of them), or a list of (query, topic) tuples if you want to
    mix e.g. "general" for company/founder lookups and "news" for a
    recent-funding-news query.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Normalize to (query, topic) pairs
    normalized = [
        q if isinstance(q, tuple) else (q, "general")
        for q in queries
    ]

    all_results = {}

    def run_one(q: str, topic: str):
        try:
            return q, search_web(q, client, topic=topic)
        except Exception as e:
            return q, [{"title": "Search failed", "url": "", "content": str(e)}]

    with ThreadPoolExecutor(max_workers=len(normalized)) as executor:
        futures = [executor.submit(run_one, q, topic) for q, topic in normalized]
        for future in as_completed(futures):
            q, results = future.result()
            all_results[q] = results

    return all_results
