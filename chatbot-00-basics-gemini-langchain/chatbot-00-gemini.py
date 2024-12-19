import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Authenticate with Google Generative AI
genai.configure(api_key=google_api_key)

# Function to handle chat with GenAI
def chat_with_genai(user_input):
    """Interact with Google Generative AI for a conversation."""
    try:
        # Initialize the model (using the latest "gemini-1.5-flash")
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Start a chat session (using a basic history)
        chat = model.start_chat(
            history=[
                {"role": "user", "parts": "Hello!"},
                {"role": "model", "parts": "Hi there! How can I assist you today?"}
            ]
        )
        
        # Send the user's message and get the response
        response = chat.send_message(user_input)
        return response.text

    except Exception as e:
        return f"Error communicating with Google GenAI: {str(e)}"

# Function to generate a study plan
def generate_study_plan(subjects, hours_per_day, exam_dates):
    plan = "Study Plan:\n"
    plan += f"Subjects: {', '.join(subjects)}\n"
    plan += f"Hours per day: {hours_per_day}\n"
    plan += f"Exam in: {exam_dates}\n\n"
    for day in range(1, len(subjects) + 1):
        plan += f"Day {day}: Focus on {subjects[day - 1]} for {hours_per_day // len(subjects)} hours.\n"
    return plan

# Streamlit UI
st.title("Agentic Study Plan Chatbot (Powered by Google GenAI)")
st.sidebar.header("Chatbot Options")

# Get user input
user_input = st.text_input("Ask your question or describe your study needs:")

if user_input:
    # Process the conversation
    if "study plan" in user_input.lower():
        st.write("Generating a study plan...")
        subjects = st.text_input("Enter subjects (comma separated):").split(',')
        hours_per_day = int(st.text_input("Enter hours per day:"))
        exam_dates = st.text_input("Enter exam date(s) or time frame:")
        study_plan = generate_study_plan(subjects, hours_per_day, exam_dates)
        st.text(study_plan)
    else:
        response = chat_with_genai(user_input)
        st.text(response)
