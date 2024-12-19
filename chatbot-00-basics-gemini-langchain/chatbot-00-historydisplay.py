import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Authenticate with Google Generative AI
genai.configure(api_key=google_api_key)

# Initialize chat history as a list of messages
chat_history = [
    {"role": "user", "parts": "Hello, act like a education counselor and study planner and teacher during this conversation."},
    {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    {"role": "user", "parts": "If the user doesn't provide context or their background, assume a basic middle school or lower background and provide response based on that. also tell about your assumption and say if you want a more customized response then please provide your background or whatever you need to know but keep it simple. "}
    

]
# Function to handle chat with GenAI using manual chat history
def chat_with_genai(user_input):
    """Interact with Google Generative AI using chat history stored in a list."""
    try:
        # Initialize the model (using "gemini-1.5-flash")
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Start chat with the history (no memory buffer, just the list)
        chat = model.start_chat(history=chat_history)
        
        # Send the user's message and get the response
        response = chat.send_message(user_input)
        
        # Store the user input and model response in chat history
        chat_history.append({"role": "user", "parts": user_input})
        chat_history.append({"role": "model", "parts": response.text})
        
        return response.text

    except Exception as e:
        return f"Error communicating with Google GenAI: {str(e)}"

# Streamlit UI Layout
st.title("Agentic Study Plan Chatbot (Powered by Google GenAI)")
st.sidebar.header("Hello! I am your Frienldy Study Planner.")
# Add a description under the sidebar header
st.sidebar.markdown("""
Welcome to your personal study planner! 🎓

I'm here to help you create flexible and dynamic study plans based on your needs. Whether you're preparing for exams, managing multiple subjects, or just looking for a better way to organize your time, I can assist you every step of the way.

Simply tell me about your study goals, subjects, or any other preferences, and I'll provide helpful advice or generate a study schedule tailored just for you. 📝

Let's get started by asking your study-related questions or describing your needs!
""")
# Create two columns: one for chat history and one for user input
col1, col2 = st.columns([1, 3])  # Adjusting column width (1:3 ratio)

# Display chat history in the left column
with col1:
    st.header("Chat History")
    for message in chat_history:
        if message["role"] == "user":
            st.markdown(f"**You**: {message['parts']}")
        else:
            st.markdown(f"**Bot**: {message['parts']}")

# Get user input in the right column
with col2:
    user_input = st.text_input("Ask your question or describe your study needs:")
    if user_input:
        # Process the conversation and generate a study plan dynamically
        response = chat_with_genai(user_input)
        st.text(response)  # Display the model's response below the input field

