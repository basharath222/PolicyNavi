import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
from database import init_db, save_user_profile, get_user_profile, save_chat_message, get_chat_history

# Initialize the SQLite database tables
init_db()

load_dotenv()

# --- 1. CACHED RESOURCE LOADING ---
@st.cache_resource
def load_models():
    """Loads the AI models and DB once and keeps them in memory."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("API Key not found! Please check your .env file.")
    else:
        genai.configure(api_key=api_key)
    
    # Using the stable 1.5-flash model
    llm_model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    db_client = chromadb.PersistentClient(path="./policynav_db")
    db_collection = db_client.get_or_create_collection(name="tamilnadu_schemes")
    
    return embed_model, llm_model, db_collection

embedding_model, model_gemini, collection = load_models()

# --- 2. CHAT LOGIC ---
def get_ai_response(user_query, history):
    # RAG Search in your Mass Scraped DB
    query_vector = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    context = "\n".join(results['documents'][0])
    
    # Building the Chat Prompt with Memory context
    # We pass the last few messages to the AI so it remembers the conversation
    full_prompt = f"""
    You are PolicyNav, an AI Civic Consultant for Tamil Nadu.
    
    CONTEXT FROM GOVT RECORDS:
    {context}
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUESTION:
    {user_query}
    
    INSTRUCTIONS:
    - If the user asks for a roadmap, provide one with Compliance, Schemes, and Documents.
    - Always include Registration URLs, Contact Numbers, and Office Addresses if available in the context.
    - If the user asks a follow-up, answer based on the context and previous chat history.
    """
    return model_gemini.generate_content(full_prompt).text

# --- 3. UI LAYOUT ---
st.title("📜 PolicyNav: AI Civic Consultant")
st.markdown("Interactive guidance for Tamil Nadu government policies.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Format history for the AI
        chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
        
        with st.spinner("Consulting government records..."):
            response = get_ai_response(prompt, chat_history_text)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})