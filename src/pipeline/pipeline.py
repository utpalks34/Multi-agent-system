import re

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    revise_chain,
)

MAX_SCRAPE_URLS = 3
MIN_ACCEPTABLE_SCORE = 7
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def _text(content) -> str:
    """Normalise message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _tool_outputs(result: dict) -> str:
    """Concatenate raw ToolMessage outputs from an agent run."""
    return "\n\n".join(
        _text(m.content) for m in result["messages"] if getattr(m, "type", "") == "tool"
    )


def _extract_urls(text: str, limit: int) -> list:
    seen, urls = set(), []
    for u in _URL_RE.findall(text):
        u = u.rstrip(".,;")
        if u not in seen:
            seen.add(u)
            urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def _parse_score(feedback: str):
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback)
    return float(m.group(1)) if m else None


def _banner(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def run_research_pipeline(topic: str, on_step=None) -> dict:
    """Run search -> read -> write -> critic. `on_step(name, state)` is called as each step starts/finishes."""
    def notify(name):
        if on_step:
            on_step(name, state)

    topic = topic.strip()
    if not topic:
        raise ValueError("Topic must not be empty.")

    state = {"topic": topic}
    notify("start")

    # step 1 - search agent
    _banner("step 1 - search agent is working ...")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    raw_search = _tool_outputs(search_result)  # untouched tool output keeps every URL
    final_search = _text(search_result["messages"][-1].content)
    state["search_results"] = final_search or raw_search
    print("\nsearch result:\n", state["search_results"])
    notify("search")

    # step 2 - reader agent (scrapes the top URLs, not just one)
    _banner("step 2 - Reader agent is scraping top resources ...")
    urls = _extract_urls(raw_search or final_search, MAX_SCRAPE_URLS)
    state["urls"] = urls

    if urls:
        url_list = "\n".join(f"- {u}" for u in urls)
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Topic: {topic}\n\nScrape each of these URLs and summarise the relevant "
                f"content from each:\n{url_list}",
            )]
        })
        scraped = _text(reader_result["messages"][-1].content)
        if not scraped.strip():
            scraped = _tool_outputs(reader_result)
        state["scraped_content"] = scraped or "No content could be scraped."
    else:
        state["scraped_content"] = "No URLs were available to scrape."
    print("\nscraped content:\n", state["scraped_content"])
    notify("reader")

    # step 3 - writer chain
    _banner("step 3 - Writer is drafting the report ...")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
    print("\nFinal Report:\n", state["report"])
    notify("writer")

    # step 4 - critic
    _banner("step 4 - critic is reviewing the report ...")
    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    state["score"] = _parse_score(state["feedback"])
    print("\ncritic report:\n", state["feedback"])
    notify("critic")

    # step 5 - one revision pass if the critic scored it low
    if state["score"] is not None and state["score"] < MIN_ACCEPTABLE_SCORE:
        _banner("step 5 - score below threshold, revising the report ...")
        state["draft_report"] = state["report"]
        state["report"] = revise_chain.invoke({
            "topic": topic,
            "research": research_combined,
            "report": state["draft_report"],
            "feedback": state["feedback"],
        })
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        state["score"] = _parse_score(state["feedback"])
        print("\nRevised Report:\n", state["report"])
        print("\nnew critic report:\n", state["feedback"])
        notify("critic")

    return state
