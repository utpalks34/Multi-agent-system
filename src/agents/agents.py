from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# Model Initialization
_groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
if not _groq_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=_groq_key,
    temperature=0,
    max_retries=3,
    timeout=90,
)

SEARCH_SYSTEM_PROMPT = (
    "You are a web research assistant. Use the web_search tool to find recent, reliable "
    "information on the user's topic. Run one or two searches if needed. In your final answer, "
    "list every relevant result exactly as returned (Title, URL, Snippet) so no URL is lost."
)

READER_SYSTEM_PROMPT = (
    "You are a content extraction assistant. Use the scrape_url tool on EACH URL the user gives "
    "you (one call per URL). If a scrape fails, move on to the next URL. In your final answer, "
    "summarise the useful facts from each page under a heading containing its URL."
)


# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()


#revision chain (used when the critic scores the report low)

revise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer who improves reports using editor feedback."),
    ("human", """Rewrite the report below so it addresses every point in the critic's feedback.
Keep the same structure (Introduction, Key Findings, Conclusion, Sources), stay factual, and
only use information present in the report or the research notes.

Topic: {topic}

Research Notes:
{research}

Current Report:
{report}

Critic Feedback:
{feedback}

Return only the improved report."""),
])

revise_chain = revise_prompt | llm | StrOutputParser()
