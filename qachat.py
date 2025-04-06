import streamlit as st
import google.generativeai as genai

# ✅ Configure Gemini with Streamlit Cloud Secret
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ✅ Load Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash-002")
chat = model.start_chat(history=[])

# ✅ Get Gemini response (non-streaming for speed + stability)
def get_gemini_response(question):
    try:
        response = chat.send_message(question)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {e}"

# ✅ Streamlit Page Setup
st.set_page_config(page_title="Startup Buddy Demo", page_icon="🚀", layout="centered")

# ✅ Custom CSS for styling
st.markdown("""
    <style>
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

# ✅ Page Header
st.markdown("<h1 style='text-align: center;'>🤖 Startup Buddy - Q&A Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Ask me anything related to your startup ideas, challenges, or funding!")

# ✅ Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ✅ Input Section
user_input = st.text_input("Ask your question below 👇", key="input")
submit = st.button("Ask the question")

# ✅ Response Processing
if submit and user_input:
    st.session_state["chat_history"].append(("you", user_input))
    with st.spinner("Thinking..."):
        response_text = get_gemini_response(user_input)
    st.session_state["chat_history"].append(("Bot", response_text))

# ✅ Chat History Display
if st.session_state["chat_history"]:
    st.subheader("🕓 Chat History")
    for role, text in reversed(st.session_state["chat_history"]):
        style = "user" if role == "you" else "bot"
        st.markdown(
            f"<div class='chat-bubble {style}'><b>{role.capitalize()}:</b> {text}</div>",
            unsafe_allow_html=True,
        )
