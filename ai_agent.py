# 1: Set up API keys for Groq and Tavily

import os

# Load environment variables from .env file
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
TAVILY_API_KEY=os.environ.get('TAVILY_API_KEY')
OEPNAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# 2: Set up the LLMs and tools
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults


# 3: Set up AI Agent with the LLMs and tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage


def call_agent(llm_model, query, allow_search, system_prompt, provider):
    if provider == "groq":
        llm = ChatGroq(model=llm_model)
    elif provider == "openAI":
        llm = ChatOpenAI(model=llm_model)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create the agent with the provided LLM and tools
    agent = create_react_agent (
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

    return ai_messages[-1].content if ai_messages else "No response from AI agent."


