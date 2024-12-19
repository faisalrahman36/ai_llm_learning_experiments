import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure API key
genai.configure(api_key=google_api_key)

# Initialize the LLM (using the free version - "gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Streamlit UI Layout
st.title("Agentic Study Plan Chatbot (Powered by Google GenAI)")
st.sidebar.header("Hello! I am your Friendly Study Planner.")

# Add a description under the sidebar header
st.sidebar.markdown("""
Welcome to your personal study planner! 🎓

I'm here to help you create flexible and dynamic study plans based on your needs. Whether you're preparing for exams, managing multiple subjects, or just looking for a better way to organize your time, I can assist you every step of the way.

Simply tell me about your study goals, subjects, or any other preferences, and I'll provide helpful advice or generate a study schedule tailored just for you. 📝

Let's get started by asking your study-related questions or describing your needs!
""")

# Create two columns: one for chat history and one for user input
col1, col2 = st.columns([1, 3])  # Adjusting column width (1:3 ratio)

# Initialize chat history as an empty list
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Define instructions for the model to follow
def generate_instructions():
    return """
    You are an educational assistant. You can act as a teacher, study planner, or student counselor.
    Assume the user is a middle schooler unless the user provides other context related to their education or study plan.
    Always be patient, clear, and supportive in your responses. 
    When offering study plans or suggestions, keep in mind that the user might not be familiar with advanced terminology.
    """

# Display chat history in the left column
with col1:
    st.header("Chat History")
    for message in st.session_state['history']:
        if message['role'] == 'user':
            st.markdown(f"**You**: {message['content']}")
        else:
            st.markdown(f"**Bot**: {message['content']}")

# Get user input in the right column
with col2:
    user_input = st.text_input("Ask your question or describe your study needs:")
    if user_input:
        # Add instruction context at the beginning of the conversation
        instructions = generate_instructions()
        
        # Append user message to chat history
        st.session_state['history'].append({'role': 'user', 'content': user_input})

        # Combine instructions with user input for context
        full_message = f"{instructions}\nUser: {user_input}"

        # Send message to the model and get response
        response = llm.invoke(full_message)

        # Append bot response to chat history
        st.session_state['history'].append({'role': 'bot', 'content': response.content})

        # Display the model's response
        st.text(response.content)
