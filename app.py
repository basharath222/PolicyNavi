import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
import sys

load_dotenv()

# --- DISABLE STREAMLIT FILE WATCHER ---
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PolicyNav - Indian Scheme Advisor", 
    page_icon="🎯",
    layout="wide"
)

# --- CONSTANTS ---
DB_PATH = "./policynav_db"

# --- Windows event loop fix ---
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'profile' not in st.session_state:
    st.session_state.profile = {}

# --- LOAD MODELS AND DATABASE ---
@st.cache_resource
def load_models_and_db():
    """Load all models and database connection"""
    # Load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Connect to database
    db_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get collection and verify it exists
    try:
        collection = db_client.get_collection(name="indian_schemes")
        count = collection.count()
        return embed_model, llm_model, collection, count
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.info("Please run init_db.py first to create the database.")
        return embed_model, llm_model, None, 0

# Load everything
embed_model, gemini_model, collection, scheme_count = load_models_and_db()

# --- ENHANCED PROMPT FUNCTION ---
def get_enhanced_prompt(query, context_chunks, user_profile, user_state):
    """Generate prompt with user context and state filtering"""
    
    formatted_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant schemes found."
    
    profile_text = "\n".join([f"- **{k}:** {v}" for k, v in user_profile.items() if v]) if user_profile else "No profile provided"
    
    prompt = f"""You are PolicyNav, an expert assistant on Indian government schemes. Your task is to provide accurate, helpful information based ONLY on the retrieved context.

## 👤 USER CONTEXT
- **State:** {user_state if user_state else 'Not specified'}
- **Category:** {user_profile.get('category', 'Not specified')}
- **Education:** {user_profile.get('education', 'Not specified')}
- **Employment:** {user_profile.get('employment', 'Not specified')}
- **Age:** {user_profile.get('age', 'Not specified')}
- **Gender:** {user_profile.get('gender', 'Not specified')}
- **Income:** {user_profile.get('income', 'Not specified')}

## 🔍 USER QUERY
{query}

## 📚 RETRIEVED SCHEMES
{formatted_context}

## 🎯 RESPONSE INSTRUCTIONS

### 1. STATE FILTERING (CRITICAL)
- User is from: **{user_state}**
- ONLY show schemes for {user_state} OR All-India schemes
- NEVER recommend schemes from other states

### 2. INCLUSIVE MATCHING
- "Post-Matric" = includes 12th pass and above
- "SC/ST/OBC" = includes the user's category
- "Professional courses" = includes engineering
- "Scholarship" = financial aid for students

### 3. RESPONSE FORMAT
For EACH scheme, provide:
### 🏷️ Scheme Name
**📝 Details:** 
**✅ Eligibility:** 
**💰 Benefits:** 
**📄 Application Process:** 
**📑 Documents Required:** 
**🔗 Source URL:** 

## 💬 YOUR RESPONSE:"""
    
    return prompt

# --- EMAIL FUNCTION ---
def send_email(to_email, subject, body):
    try:
        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        
        if not sender or not password:
            return False, "Email credentials not set"
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to_email
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, str(e)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 PolicyNav")
    st.caption("AI-Powered Scheme Advisor")
    
    # Database status
    if collection is not None:
        st.success(f"✅ {scheme_count} schemes loaded")
    else:
        st.error("Database not found")
        st.stop()
    
    st.divider()
    
    # User profile
    with st.expander("📝 Your Profile", expanded=True):
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        education = st.selectbox("Education", ["", "10th", "12th", "Graduate", "Diploma", "ITI"])
        employment = st.selectbox("Employment", ["", "Unemployed", "Employed", "Student", "Business", "Farmer", "Retired"])
        income = st.selectbox("Income", ["", "Below ₹1L", "₹1-2.5L", "₹2.5-5L", "Above ₹5L"])
        state = st.selectbox("State", [
            "", "Tamil Nadu", "Kerala", "Karnataka", "Maharashtra", "Delhi", 
            "Uttar Pradesh", "Gujarat", "Punjab", "Himachal Pradesh", "West Bengal"
        ])
        category = st.selectbox("Category", ["", "General", "SC", "ST", "OBC", "EWS"])
        
        if st.button("💾 Save Profile", use_container_width=True, type="primary"):
            st.session_state.profile = {
                "age": age,
                "gender": gender,
                "education": education,
                "employment": employment,
                "income": income,
                "state": state,
                "category": category
            }
            st.success("✅ Profile saved!")
    
    st.divider()
    
    # Email summary
    with st.expander("📧 Email Summary", expanded=False):
        email = st.text_input("Your Email")
        if st.button("📨 Send Summary", use_container_width=True):
            if email and st.session_state.messages:
                summary = "PolicyNav Chat Summary\n\n"
                summary += f"Date: {datetime.now()}\n"
                summary += f"Profile: {st.session_state.profile}\n\n"
                for msg in st.session_state.messages[-10:]:
                    summary += f"{msg['role'].upper()}: {msg['content']}\n\n"
                
                success, msg = send_email(email, "PolicyNav Summary", summary)
                if success:
                    st.success(msg)
                else:
                    st.error(f"Email failed: {msg}")
    
    st.divider()
    
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("📜 PolicyNav - Indian Government Scheme Advisor")
st.caption("Ask about schemes • I'll find what you're eligible for")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about schemes..."):
    if collection is None:
        st.error("Database not available. Please check initialization.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching schemes..."):
            
            try:
                # Get user state
                user_state = st.session_state.profile.get('state', '')
                
                # Create search query
                profile_text = " ".join([str(v) for v in st.session_state.profile.values() if v])
                search_query = f"{user_state} {prompt} {profile_text}"
                
                # Search database
                query_vector = embed_model.encode(search_query).tolist()
                results = collection.query(
                    query_embeddings=[query_vector], 
                    n_results=10,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results['documents'] and results['documents'][0]:
                    # Filter results
                    filtered_docs = []
                    filtered_metas = []
                    
                    for i, doc in enumerate(results['documents'][0][:5]):
                        metadata = results['metadatas'][0][i]
                        
                        # Simple state check
                        doc_lower = doc.lower()
                        if user_state:
                            if user_state.lower() in doc_lower or 'all india' in doc_lower:
                                filtered_docs.append(doc)
                                filtered_metas.append(metadata)
                        else:
                            filtered_docs.append(doc)
                            filtered_metas.append(metadata)
                    
                    if filtered_docs:
                        # Generate response
                        enhanced_prompt = get_enhanced_prompt(
                            query=prompt,
                            context_chunks=filtered_docs,
                            user_profile=st.session_state.profile,
                            user_state=user_state
                        )
                        
                        response = gemini_model.generate_content(enhanced_prompt)
                        response_text = response.text
                        
                        # Add scheme names
                        scheme_names = [m.get('scheme_name', 'Scheme') for m in filtered_metas]
                        if scheme_names:
                            response_text += f"\n\n---\n**📌 Schemes found:** {', '.join(scheme_names)}"
                        
                        st.markdown(response_text)
                    else:
                        st.info("No schemes found for your state. Showing general results:")
                        for i, doc in enumerate(results['documents'][0][:3]):
                            scheme_name = results['metadatas'][0][i].get('scheme_name', 'Scheme')
                            st.markdown(f"- **{scheme_name}**")
                        response_text = "No state-specific schemes found."
                else:
                    response_text = "I couldn't find any schemes. Try different keywords."
                    st.warning(response_text)
                    
            except Exception as e:
                response_text = f"Search error: {str(e)[:100]}"
                st.error(response_text)
            
            # Save response
            st.session_state.messages.append({"role": "assistant", "content": response_text})