
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
import os
import google.generativeai as genai
import sys
import time

# Add the backend directory to the Python path if running from frontend
import sys
# from pathlib import Path
# backend_path = Path(__file__).parent
# if not str(backend_path) in sys.path:
#     sys.path.append(str(backend_path))

# Function to get API keys from environment or Streamlit secrets
def get_api_key(key_name):
    # First try environment variables
    value = os.getenv(key_name)
    
    # If not found, try Streamlit secrets
    if not value and 'secrets' in dir(st):
        try:
            value = st.secrets[key_name]
        except:
            pass
            
    return value

# Streamlit Page Configuration
st.set_page_config(page_title="Startup Buddy Demo", page_icon="🚀", layout="centered")

# Custom CSS
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
        .error-message {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header and Input
st.markdown("<h1 style='text-align: center;'>🤖 Startup Buddy - Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask me anything related to your startup ideas, challenges, or funding!")

# Check for environment variables
pinecone_api_key = get_api_key("PINECONE_API_KEY")
pinecone_index_name = get_api_key("PINECONE_INDEX_NAME")
google_api_key = get_api_key("GOOGLE_API_KEY")

# Display config status in sidebar
with st.sidebar:
    st.title("Configuration Status")
    
    # Check Google API key
    if google_api_key:
        st.success("✅ Google API key is configured")
    else:
        st.error("❌ Google API key is missing")
        
    # Check Pinecone configuration
    if pinecone_api_key:
        st.success("✅ Pinecone API key is configured")
    else:
        st.error("❌ Pinecone API key is missing")
        
    if pinecone_index_name:
        st.success(f"✅ Pinecone index name is configured: {pinecone_index_name}")
    else:
        st.error("❌ Pinecone index name is missing")
    
    # Add configuration form if needed
    if not (google_api_key and pinecone_api_key and pinecone_index_name):
        with st.expander("Configure API Keys"):
            with st.form("api_keys_form"):
                if not google_api_key:
                    new_google_api_key = st.text_input("Google API Key", type="password")
                if not pinecone_api_key:
                    new_pinecone_api_key = st.text_input("Pinecone API Key", type="password")
                if not pinecone_index_name:
                    new_pinecone_index_name = st.text_input("Pinecone Index Name")
                
                submit_button = st.form_submit_button("Save Configuration")
                
                if submit_button:
                    # Here you could save these to .env file or st.session_state
                    st.success("Configuration saved. Please restart the app.")
                    time.sleep(2)
                    st.experimental_rerun()

# Configure Gemini
try:
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-002")
    chat = model.start_chat(history=[])
except Exception as e:
    st.error(f"Error configuring Gemini: {str(e)}")
    if not google_api_key:
        st.error("Google API key is missing. Please check your configuration.")
    chat = None

# Import Pinecone utilities - with proper error handling for the new structure
try:
    # Try direct import first (if in the same directory)
    try:
        from pinecone_utils import store_chat, search_similar
    except ImportError:
        # If direct import fails, try relative import (from backend folder)
        from backend.pinecone_utils import store_chat, search_similar
    
    pinecone_available = True
    st.sidebar.success("✅ Pinecone integration initialized successfully")
except Exception as e:
    st.sidebar.error(f"❌ Pinecone error: {str(e)}")
    pinecone_available = False
    
    # Dummy functions if Pinecone is not available
    def store_chat(question, response):
        return False
    
    def search_similar(query, top_k=3):
        return []

# Function to get Gemini response
def get_gemini_response(question):
    if chat is None:
        return None
    
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        st.error(f"Error getting response from Gemini: {str(e)}")
        return None

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input field
input_text = st.text_input("Ask your question below 👇", key="input")
submit = st.button("Ask the question")

# Process input
if submit and input_text and chat is not None:
    st.session_state['chat_history'].append(("you", input_text))
    
    # Still perform similarity search in the background but don't display results
    if pinecone_available:
        with st.spinner("Searching knowledge base..."):
            # This line runs the search but results are not displayed
            similar_docs = search_similar(input_text)
            # You can optionally log this information in the sidebar if needed for debugging
            if similar_docs:
                st.sidebar.info(f"Found {len(similar_docs)} similar entries in knowledge base")

    # Get Gemini response
    with st.spinner("Generating response..."):
        response = get_gemini_response(input_text)
        if response:
            st.subheader("💬 Response:")
            full_response = ""
            for chunk in response:
                if chunk and hasattr(chunk, 'text') and chunk.text is not None:
                    st.write(chunk.text)
                    full_response += chunk.text
            
            if full_response:
                st.session_state['chat_history'].append(("Bot", full_response))
                
                # Store chat in Pinecone but don't show the visual feedback
                if pinecone_available:
                    store_result = store_chat(input_text, full_response)
                    # You can log this in the sidebar for debugging if needed
                    if store_result:
                        st.sidebar.success("Conversation stored in memory")
                    else:
                        st.sidebar.warning("Failed to store conversation")
            else:
                st.error("Received empty response from Gemini")

# Display only the user-bot chat history, not the similarity search results
st.subheader("💬 Chat History")
for role, text in reversed(st.session_state['chat_history']):
    style = "user" if role == "you" else "bot"
    if text:  # Ensure text is not None
        st.markdown(f"<div class='chat-bubble {style}'><b>{role.capitalize()}:</b> {text}</div>", unsafe_allow_html=True)




