import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY"))

# Set up Gemini Free API Key from environment
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your environment variables.")
    st.stop()

# Set up Tavily API Key from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    st.error("Tavily API key not found. Please set TAVILY_API_KEY in your environment variables.")
    st.stop()

# Define State for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    human_requested: bool
    human_response_ready: bool
    human_response: str
    needs_clarification: bool
    search_requested: bool

graph_builder = StateGraph(State)

# Initialize Gemini Free Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_KEY)

# Define AI Agent Node
def ai_agent(state: State):
    """Handles student queries and decides if human help is needed."""
    response = llm.invoke(state["messages"])
    
    if hasattr(response, "content"):
        response_content = response.content
    else:
        response_content = str(response)
    
    if "human assistance" in response_content.lower():
        state["human_requested"] = True
    
    if "search online" in response_content.lower() or "find on the internet" in response_content.lower():
        state["search_requested"] = True
    
    if "can you clarify" in response_content.lower() or "need more information" in response_content.lower():
        state["needs_clarification"] = True
    
    state["messages"].append({"role": "ai", "content": response_content})
    return state

graph_builder.add_node("ai_agent", ai_agent)

# Define Internet Search Tool
search_tool = TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)
tool_node = ToolNode(tools=[search_tool])
graph_builder.add_node("search", tool_node)

def human_assist(state: State):
    return state

graph_builder.add_node("human_assist", human_assist)

def clarification_needed(state: State):
    return state

graph_builder.add_node("clarification", clarification_needed)

# Routing Logic
def route_logic(state: State):
    if state["human_requested"]:
        return "human_assist"
    if state["search_requested"]:
        return "search"
    if state["needs_clarification"]:
        return "clarification"
    return END

graph_builder.add_conditional_edges("ai_agent", route_logic, {"human_assist": "human_assist", "search": "search", "clarification": "clarification", END: END})

graph_builder.add_edge(START, "ai_agent")
graph_builder.add_edge("human_assist", "ai_agent")
graph_builder.add_edge("search", "ai_agent")
graph_builder.add_edge("clarification", "ai_agent")

graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📚 AI Study Planner & Student Counselor")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "human_response_ready" not in st.session_state:
    st.session_state["human_response_ready"] = False
if "human_response" not in st.session_state:
    st.session_state["human_response"] = ""
if "human_requested" not in st.session_state:
    st.session_state["human_requested"] = False
if "search_requested" not in st.session_state:
    st.session_state["search_requested"] = False

with st.sidebar:
    st.header("🗂️ Chat History")
    for msg in st.session_state["chat_history"]:
        role = "👤 User" if msg["role"] == "user" else "🤖 AI" if msg["role"] == "ai" else "👨‍🏫 Human Assistant"
        st.write(f"{role}: {msg['content']}")
    
    st.header("📖 How to Use")
    st.markdown("""
    - 💬 Ask a study-related question.
    - 🤖 AI will respond.
    - 👨‍🏫 Click "Fetch Human Response" if AI needs help.
    - 🌍 AI will search online if required.
    - ✨ Your conversation history is saved.
    """)

user_input = st.text_input("Ask a study-related question:")
if st.button("Submit AI Query") and user_input:
    state = {
        "messages": st.session_state["chat_history"] + [{"role": "user", "content": user_input}],
        "human_requested": False,
        "search_requested": False,
        "needs_clarification": False,
        "human_response_ready": st.session_state["human_response_ready"],
        "human_response": st.session_state["human_response"]
    }
    state = ai_agent(state)
    st.session_state["chat_history"].append(state["messages"][-1])
    st.write("💡 AI Response:", state["messages"][-1]["content"])
    st.session_state["human_requested"] = state["human_requested"]
    st.session_state["search_requested"] = state["search_requested"]

if st.session_state["search_requested"]:
    search_results = search_tool.invoke(user_input)
    st.write("🌍 Internet Search Results:")
    for idx, result in enumerate(search_results):
        st.write(f"{idx+1}. {result['title']} - {result['link']}")
        st.write(result["snippet"])
    st.session_state["search_requested"] = False

if st.session_state["human_requested"] and not st.session_state["human_response_ready"]:
    if st.button("Fetch Human Response"):
        st.session_state["human_response_ready"] = True

if st.session_state["human_response_ready"]:
    human_response = st.text_area("Enter human response here:", value=st.session_state["human_response"])
    if st.button("Submit Human Response"):
        st.session_state["human_response"] = human_response
        st.session_state["chat_history"].append({"role": "human", "content": human_response})
        st.session_state["human_response_ready"] = False
        st.session_state["human_requested"] = False
        st.write("👨‍🏫 Human Assistant:", human_response)
