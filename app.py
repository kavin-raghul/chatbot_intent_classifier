import streamlit as st
import sys
import os

# Add the project root to sys.path so the module can be found
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict_intent import IntentPredictor

st.set_page_config(page_title="Intent Classification Chatbot", page_icon="🤖")

st.title("🤖 Intent Classification Chatbot")
st.markdown("A simple NLP chatbot that classifies your messages into predefined intents using TF-IDF and classical ML algorithms.")

@st.cache_resource
def load_predictor():
    try:
        return IntentPredictor()
    except FileNotFoundError:
        return None

predictor = load_predictor()

if predictor is None:
    st.error("Model files not found! Please run `python src/train_model.py` to train your intents model first.")
    st.stop()

assert predictor is not None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's on your mind?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = predictor.get_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.title("About")
st.sidebar.info(
    "This is an end-to-end NLP project utilizing TF-IDF and traditional Machine Learning classification models (Naive Bayes, SVM, Logistic Regression)."
)
