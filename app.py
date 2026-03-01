import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. CACHED RESOURCE LOADING ---

@st.cache_resource
def load_models():
    """Loads the AI models and DB once and keeps them in memory."""
    print("Initializing Models (This should only run ONCE)...")
    
    # 1. Load Embedding Model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("API Key not found! Please check your .env file.")
    else:
        genai.configure(api_key=api_key)
    
    llm_model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    # 3. Connect to ChromaDB
    db_client = chromadb.PersistentClient(path="./policynav_db")
    db_collection = db_client.get_or_create_collection(name="tamilnadu_schemes")
    
    return embed_model, llm_model, db_collection

# Fetch the global instances (Fast after the first run)
embedding_model, model_gemini, collection = load_models()

# --- 2. REST OF YOUR APP LOGIC ---
# (The rest of your app.py remains the same)

# --- 2. LOGIC FUNCTIONS ---
def get_intent(query):
    prompt = f"Extract entities from this query in strict JSON: '{query}'. Fields: Business_Type, Location, Category."
    response = model_gemini.generate_content(prompt)
    # Basic cleaning to handle potential markdown in response
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

def get_roadmap(query):
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=4)
    
    context = "\n".join(results['documents'][0])
    # Store sources in session state so the UI can access them later
    st.session_state.sources = results['metadatas'][0] 
    
   
    prompt = f"""
    Using ONLY the context below, provide a highly detailed roadmap for: {query}.
    CONTEXT: {context}

    IF AVAILABLE in the context, you MUST include:
    - Direct Registration/Application URLs.
    - Contact Numbers and Department Email IDs.
    - Physical Office Addresses (e.g., MSME-DFO Guindy).

    Format the output with bold headers and bullet points for easy reading.
    """
    return model_gemini.generate_content(prompt).text

# --- 3. UI LAYOUT ---
st.title("📜 PolicyNav: TN Civic Consultant")
st.markdown("Helping you navigate Tamil Nadu government policies and schemes with AI.")

# Initialize session state for the workflow
if "step" not in st.session_state:
    st.session_state.step = 1

# STEP 1: SMART INTAKE
if st.session_state.step == 1:
    st.subheader("How can we help you today?")
    user_input = st.text_input("e.g., I want to start a small tea stall in Chennai as an unemployed youth.")
    
    if st.button("Generate My Roadmap"):
        if user_input:
            with st.spinner("Analyzing Intent & Searching Records..."):
                st.session_state.intent = get_intent(user_input)
                st.session_state.roadmap = get_roadmap(user_input)
                st.session_state.step = 2
                st.rerun()

# STEP 2: STRUCTURED DASHBOARD
elif st.session_state.step == 2:
    # Sidebar Profile Card
    with st.sidebar:
        st.header("👤 Your Profile")
        st.write(f"**Business:** {st.session_state.intent.get('Business_Type', 'N/A')}")
        st.write(f"**Location:** {st.session_state.intent.get('Location', 'N/A')}")
        if st.button("Reset / New Search"):
            st.session_state.step = 1
            st.rerun()

    # Main Output Cards
    st.success("✅ Roadmap Successfully Generated!")
    
    # Using Tabs for the "Non-Normal Chat" feel
    tab1, tab2 = st.tabs(["📋 Your Custom Roadmap", "📂 Source Documents"])
    
    with tab1:
        st.markdown(st.session_state.roadmap)
    
    with tab2:
        st.info("These recommendations are based on verified TN Government Records & MSME Booklets.")
        
        if "sources" in st.session_state:
            st.subheader("Reference Materials Used:")
            for meta in st.session_state.sources:
                # Create a nice card for each source found in your policynav_db
                with st.expander(f"📄 {meta.get('scheme_name', 'Government Policy')}"):
                    st.write(f"**Section:** {meta.get('tab', 'General Info')}")
                    st.write(f"**Source Link:** {meta.get('source', 'Local Database')}")