# save as test_langsmith.py and run: python test_langsmith.py

import os
os.environ["LANGSMITH_TRACING"]  = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"]  = "your_key_here"   # ← paste your key
os.environ["LANGCHAIN_API_KEY"]  = "your_key_here"   # ← same key
os.environ["LANGSMITH_PROJECT"]  = "langgraph-production"
os.environ["LANGCHAIN_PROJECT"]  = "langgraph-production"

from langsmith import Client

client = Client()

# Test 1: can we connect?
try:
    projects = list(client.list_projects())
    print("Connected! Projects found:")
    for p in projects:
        print(f"  - {p.name}")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Test 2: send a real traced LLM call
from langchain_core.messages import HumanMessage

print("\nSending test trace...")
try:
    # This import triggers the tracer
    from langchain_groq import ChatGroq   # or ChatOpenAI etc.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY", "your_groq_key"),
    )
    resp = llm.invoke([HumanMessage(content="say hello in 3 words")])
    print(f"Response: {resp.content}")
    print("\nCheck LangSmith dashboard — you should see a trace now!")
except Exception as e:
    print(f"LLM call failed: {e}")