import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your environment variables.")
    st.stop()
if not TAVILY_API_KEY:
    st.error("Tavily API key not found. Please set TAVILY_API_KEY in your environment variables.")
    st.stop()

# Define State for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    human_requested: bool
    human_response_ready: bool
    human_response: str
    search_requested: bool

# Initialize LangGraph
graph_builder = StateGraph(State)

# Initialize Gemini AI Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_KEY)

# Initialize Tavily Search Tool
search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)

def ai_agent(state: State):
    """Handles student queries and determines if human help or search is needed."""
    user_query = state["messages"][-1]["content"].lower()
    
    if "search online" in user_query or "find on the internet" in user_query:
        state["search_requested"] = True
        return state
    
    response = llm.invoke(state["messages"])
    response_content = response.content if hasattr(response, "content") else str(response)
    
    if "human assistance" in response_content.lower():
        state["human_requested"] = True
    
    state["messages"].append({"role": "ai", "content": response_content})
    return state

graph_builder.add_node("ai_agent", ai_agent)

def search_online(state: State):
    """Fetches search results using Tavily API."""
    search_query = state["messages"][-1]["content"]
    results = search_tool.invoke(search_query)
    
    if not results:
        state["messages"].append({"role": "search", "content": "No relevant search results found."})
        return state
    
    formatted_results = "\n\n".join(
        [f"{res.get('url', 'No URL')}\n{res.get('content', 'No Content')}" for res in results]
    )
    state["messages"].append({"role": "search", "content": formatted_results})
    return state

graph_builder.add_node("search", search_online)

def human_assist(state: State):
    """Fetches human assistance when required."""
    state["human_response_ready"] = True
    return state

graph_builder.add_node("human_assist", human_assist)

def route_logic(state: State):
    if state["human_requested"]:
        return "human_assist"
    if state["search_requested"]:
        return "search"
    return END

graph_builder.add_conditional_edges("ai_agent", route_logic, {"human_assist": "human_assist", "search": "search", END: END})

graph_builder.add_edge(START, "ai_agent")
graph_builder.add_edge("human_assist", "ai_agent")
graph_builder.add_edge("search", "ai_agent")

graph = graph_builder.compile()

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📚 AI Study Planner & Student Counselor")

# Initialize session states
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

# Sidebar with chat history
with st.sidebar:
    st.header("🗂️ Chat History")
    for msg in st.session_state["chat_history"]:
        role = "👤 User" if msg["role"] == "user" else "🤖 AI" if msg["role"] == "ai" else "🌍 Search" if msg["role"] == "search" else "👨‍🏫 Human Assistant"
        st.write(f"{role}: {msg['content']}")
    
    st.header("📖 How to Use")
    st.markdown("""
    - 💬 Ask a study-related question.
    - 🤖 AI will respond.
    - 🌍 Write "search online" to trigger an internet search.
    - 👨‍🏫 Write "human assistance" and then click "Fetch Human Response" if AI is needed to take human help.
    - ✨ Your conversation history is saved.
    """)

# Main Chat Interface
user_input = st.text_input("Ask a study-related question:")
if st.button("Submit AI Query") and user_input:
    state = {
        "messages": st.session_state["chat_history"] + [{"role": "user", "content": user_input}],
        "human_requested": False,
        "search_requested": False,
        "human_response_ready": st.session_state["human_response_ready"],
        "human_response": st.session_state["human_response"]
    }
    state = ai_agent(state)
    st.session_state["chat_history"].append(state["messages"][-1])
    st.write("💡 AI Response:", state["messages"][-1]["content"])
    st.session_state["human_requested"] = state["human_requested"]
    st.session_state["search_requested"] = state["search_requested"]

# Handle Search Requests
if st.session_state["search_requested"]:
    state = search_online(state)
    st.session_state["chat_history"].append(state["messages"][-1])
    st.write("🌍 Internet Search Results:", state["messages"][-1]["content"])
    st.session_state["search_requested"] = False

# Handle Human Assistance Requests
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
