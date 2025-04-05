
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load model
model = genai.GenerativeModel("gemini-1.5-flash-002")
chat = model.start_chat(history=[])

# Function to get Gemini response (name not changed)
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Streamlit Page Configuration
st.set_page_config(page_title="Startup Buddy Demo", page_icon="🚀", layout="centered")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 12px;
        }
        .stButton>button {
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .chat-bubble {
            background-color: #e9ecef;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .user {
            text-align: right;
            color: #007BFF;
        }
        .bot {
            text-align: left;
            color: #000;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Input
st.markdown("<h1 style='text-align: center;'>🤖 Startup Buddy - Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask me anything related to your startup ideas, challenges, or funding!")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input field
input = st.text_input("Ask your question below 👇", key="input")
submit = st.button("Ask the question")

# Process user input
if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("you", input))

    st.subheader("💬 Response:")
    full_response = ""
    for chunk in response:
        st.write(chunk.text)
        full_response += chunk.text
    st.session_state['chat_history'].append(("Bot", full_response))

# Chat History Display
if st.session_state['chat_history']:
    st.subheader("🕓 Chat History")
    for role, text in reversed(st.session_state['chat_history']):
        style = "user" if role == "you" else "bot"
        st.markdown(f"<div class='chat-bubble {style}'><b>{role.capitalize()}:</b> {text}</div>", unsafe_allow_html=True)
