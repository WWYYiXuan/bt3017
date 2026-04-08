import os
import streamlit as st
import requests
from dotenv import load_dotenv
from utils import init_session_state

load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("DEEPSEEK_API_KEY")

st.title("Ask the Tutor")
st.write(
    "Ask questions about kernel trick, feature mapping, SVM, and PCA."
)

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = False

# Reuse shared app state where useful
init_session_state()

dataset_type = st.session_state.get("dataset_type", None)
mapping_name = st.session_state.get("mapping_name", None)

context_text = f"""
Current context:
- dataset: {dataset_type if dataset_type else "Not selected"}
- mapping: {mapping_name if mapping_name else "Not selected"}
"""

# Get API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY is not set. Please set it before running.")
    st.stop()

# -------------------------------
# Chat history
# -------------------------------
if "tutor_messages" not in st.session_state:
    st.session_state.tutor_messages = [
        {
            "role": "assistant",
            "content": "Hi! I’m your BT3017 tutor. Ask me anything about kernel trick, SVM, or PCA."
        }
    ]

# -------------------------------
# Show context (nice feature)
# -------------------------------
with st.expander("Current app context"):
    st.write(f"Dataset: **{dataset_type}**")
    st.write(f"Mapping: **{mapping_name}**")

# -------------------------------
# Display chat
# -------------------------------
for msg in st.session_state.tutor_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# DeepSeek API function
# -------------------------------
def ask_deepseek(messages):
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"API Error: {result}"

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# User input
# -------------------------------
user_input = st.chat_input("Ask a question...")

if user_input:
    # Add user message
    st.session_state.tutor_messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------
    # System prompt (VERY IMPORTANT)
    # -------------------------------
    system_prompt = f"""
You are a teaching assistant for BT3017.

Explain concepts clearly for undergraduate students.

Focus on:
- kernel trick
- feature mapping
- linear vs RBF SVM
- PCA

{context_text}

Rules:
- use simple explanations
- relate to visualizations in the app
- use examples like Circles, XOR, Moons
- keep answers concise
- highlight intuition
"""

    # Build message list
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.tutor_messages

    # -------------------------------
    # Get response
    # -------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask_deepseek(messages)
        st.markdown(reply)

    # Save response
    st.session_state.tutor_messages.append(
        {"role": "assistant", "content": reply}
    )

# -------------------------------
# Buttons
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear chat"):
        st.session_state.tutor_messages = [
            {
                "role": "assistant",
                "content": "Hi! I’m your BT3017 tutor. Ask me anything about kernel trick, SVM, or PCA."
            }
        ]
        st.rerun()

with col2:
    if st.button("Suggest questions"):
        st.session_state.show_suggestions = not st.session_state.show_suggestions
        if st.session_state.show_suggestions:
            st.info(
                f"""
- Why does {dataset_type} fail with a linear classifier?
- When does {mapping_name} help?
- Why is XOR not linearly separable?
- When should I use RBF instead of linear SVM?
- What is the difference between kernel trick and PCA?
"""
            )

st.info("💡 This tutor uses an AI model (DeepSeek) to explain concepts interactively.")