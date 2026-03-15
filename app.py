import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import uuid
import re
from dotenv import load_dotenv
from database import save_user_profile, get_user_profile, save_chat_message, get_chat_history

load_dotenv()

# --- 1. SESSION INITIALIZATION ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

# --- 2. MODELS & DB LOADING ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    
    # Using stable flash model
    llm_model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    db_client = chromadb.PersistentClient(path="./policynav_db")
    db_collection = db_client.get_or_create_collection(name="tamilnadu_schemes")
    return embed_model, llm_model, db_collection

embedding_model, model_gemini, collection = load_models()

# --- 3. AI LOGIC (The Interviewer) ---
def get_ai_response(user_query, history, user_profile):
    # Search Vector DB
    query_vector = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    context = "\n".join(results['documents'][0])
    
    # A. Entity Extraction (Invisible)
    extract_prompt = f"Extract age, gender, education as JSON from: '{user_query}'. If none found, return {{}}."
    extraction_raw = model_gemini.generate_content(extract_prompt).text
    try:
        match = re.search(r'\{.*\}', extraction_raw, re.DOTALL)
        if match:
            new_data = json.loads(match.group())
            if new_data:
                updated_profile = {**user_profile, **new_data}
                save_user_profile(st.session_state.user_id, updated_profile)
    except:
        pass

    # B. Generate Main Response
    full_prompt = f"""
    You are PolicyNav, the official AI Civic Consultant for Tamil Nadu.
    USER PROFILE: {user_profile}
    CONTEXT: {context}
    HISTORY: {history}
    
    INSTRUCTIONS:
    - You represent 234 verified Tamil Nadu schemes.
    - If the user wants a scheme (e.g. marriage, business) but PROFILE is missing GENDER, AGE, or EDUCATION, 
      congratulate them but ASK for the missing info politely first.
    - If you have the info, give a clear roadmap: Steps, Documents, and Office Locations (like Guindy).
    
    USER: {user_query}
    """
    return model_gemini.generate_content(full_prompt).text

# --- 4. UI LAYOUT ---
st.title("📜 PolicyNav: AI Civic Consultant")
st.markdown("#### *Your personal guide to 234 Tamil Nadu Government Schemes*")

with st.sidebar:
    st.header("📊 System Status")
    st.metric("Database Coverage", "234 TN Schemes")
    
    # Visual Profile Verification
    st.subheader("👤 Detected Profile")
    profile = get_user_profile(st.session_state.user_id)
    if profile:
        for k, v in profile.items():
            st.write(f"**{k.capitalize()}:** {v}")
    else:
        st.write("No details found yet.")

    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Load History
if "messages" not in st.session_state:
    st.session_state.messages = get_chat_history(st.session_state.user_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. CHAT INPUT ---
if prompt := st.chat_input("How can I help you today?"):
    save_chat_message(st.session_state.user_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Fetch profile for the AI
        user_profile = get_user_profile(st.session_state.user_id)
        chat_history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
        
        with st.spinner("Consulting 234 records..."):
            response = get_ai_response(prompt, chat_history_text, user_profile)
            st.markdown(response)
            save_chat_message(st.session_state.user_id, "assistant", response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})