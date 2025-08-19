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

openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

# 3: Set up AI Agent with the LLMs and tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "You are a very intelligent, creative and a friendly AI assistant." 

agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    prompt=system_prompt
)

query = "Tell me about the most beautiful place in Japan."
state = {"messages": query}
response = agent.invoke(state)

# 4: Print the response
messages = response.get("messages")
ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
print("final response: ", ai_messages[-1].content)
# print(response)


