import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

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
CSV_PATH = "cleaned_my_scheme_data_fixed.csv"
COLLECTION_NAME = "indian_schemes"

# --- Windows event loop fix ---
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'profile' not in st.session_state:
    st.session_state.profile = {}

# --- HELPER: INITIALIZE DATABASE FROM CSV ---
import gc

# --- CONSTANTS ---
DB_PATH = "./policynav_db"
CSV_PATH = "cleaned_my_scheme_data_fixed.csv"
COLLECTION_NAME = "indian_schemes"

# --- HELPER: INITIALIZE DATABASE WITH LOW MEMORY/CPU FOOTPRINT ---
def build_cloud_db(client, embed_model):
    """Builds database on first run with low memory and CPU-friendly batching."""
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Missing dataset: {CSV_PATH}")
        return None, 0

    col = client.get_or_create_collection(name=COLLECTION_NAME)
    df = pd.read_csv(CSV_PATH).fillna("")

    docs, metas, ids = [], [], []
    for idx, row in df.iterrows():
        s_name = str(row.get("scheme_name", "")).strip()
        details = str(row.get("details", "")).strip()
        benefits = str(row.get("benefits", "")).strip()
        eligibility = str(row.get("eligibility", "")).strip()
        app_proc = str(row.get("application_process", row.get("how_to_apply", ""))).strip()
        docs_req = str(row.get("documents_required", row.get("documents", ""))).strip()
        url = str(row.get("url", row.get("source_url", ""))).strip()
        state = str(row.get("state", "All-India")).strip()
        cat = str(row.get("category", "")).strip()

        chunk = (
            f"Scheme Name: {s_name}\nState: {state}\nCategory: {cat}\n"
            f"Details: {details}\nEligibility: {eligibility}\nBenefits: {benefits}\n"
            f"Application Process: {app_proc}\nDocuments Required: {docs_req}\nSource URL: {url}"
        )
        docs.append(chunk)
        metas.append({"scheme_name": s_name[:100], "state": state, "category": cat, "url": url})
        ids.append(f"s_{idx}")

    # Small batch size + garbage collection prevents Streamlit Cloud CPU throttling
    batch_sz = 64
    total = len(docs)
    progress_bar = st.progress(0, text="⚙️ Initializing scheme database...")
    
    for i in range(0, total, batch_sz):
        end = min(i + batch_sz, total)
        embs = embed_model.encode(docs[i:end], batch_size=batch_sz, show_progress_bar=False).tolist()
        col.add(documents=docs[i:end], embeddings=embs, metadatas=metas[i:end], ids=ids[i:end])
        progress_bar.progress(end / total, text=f"⚙️ Indexing schemes ({end}/{total})...")
        gc.collect()

    progress_bar.empty()
    return col, col.count()

# --- LOAD MODELS AND DATABASE ---
@st.cache_resource
def load_models_and_db():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    
    db_client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        col = db_client.get_collection(name=COLLECTION_NAME)
        count = col.count()
        if count == 0:
            raise ValueError("Empty collection")
    except Exception:
        with st.spinner("Setting up database for first-time cloud deployment..."):
            col, count = build_cloud_db(db_client, embed_model)
            if col is None:
                return embed_model, llm_model, None, 0

    return embed_model, llm_model, col, count

embed_model, gemini_model, collection, scheme_count = load_models_and_db()

# --- ENHANCED PROMPT FUNCTION ---
def get_enhanced_prompt(query, context_chunks, user_profile, user_state):
    formatted_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant schemes found."
    profile_text = "\n".join([f"- **{k}:** {v}" for k, v in user_profile.items() if v]) if user_profile else "No profile provided"
    
    prompt = f"""You are PolicyNav, an expert civic consultant on Indian government schemes. Your task is to provide accurate, actionable roadmaps based on the retrieved context.

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

## 📚 RETRIEVED SCHEMES (CONTEXT ONLY - USE THESE)
{formatted_context}

## 🎯 INSTRUCTIONS FOR MISSING PROCESS/URL FIELDS:
1. If the context has explicit "Application Process" or "Documents Required", use them verbatim.
2. If the context states that the scheme is managed by a specific department (e.g., "Backward Classes and Minorities Welfare Department, Tamil Nadu" or "Higher Education Department") but leaves the application blank:
   - Formulate standard administrative steps (e.g., "Apply via the head of the admitted institution / College Principal office" or "Visit the District Backward Classes and Minorities Welfare Office / nearest e-Sevai centre").
   - List essential documents standard for educational assistance: Community Certificate, 10th/12th Marksheets, Income Certificate, College Allotment Order, Fee Receipt, and Bank Passbook copy.
3. If "Source URL" is blank or not available in the context chunk:
   - Provide the official state or national portal relevant to this department (e.g., https://www.myscheme.gov.in, https://tnesevai.tn.gov.in, or https://scholarships.gov.in). Do not leave it blank.

## 🎯 RESPONSE STRUCTURE PER SCHEME:
### 🏷️ Scheme Name
**📝 Details:** (2-3 sentences about the scheme)

**✅ Eligibility:**
- ✓ **Matches your profile:** [explain what matches]
- ⚠️ **Note:** [explain criteria or constraints]

**💰 Benefits:** (Financial and non-financial support)

**📄 Application Process:** (Step-by-step procedural roadmap)

**📑 Documents Required:** (Bullet list of checklist items)

**🔗 Source Portal:** [Direct link or department portal]

#### Comparison Table:
Add a summary table comparing the schemes at the end.

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
    
    if collection is not None:
        st.success(f"✅ {scheme_count} schemes loaded")
    else:
        st.error("Database not found")
        st.stop()
    
    st.divider()
    
    with st.expander("📝 Your Profile", expanded=True):
        age = st.number_input("Age", 18, 100, 18)
        gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
        education = st.selectbox("Education", ["", "12th", "10th", "Graduate", "Diploma", "ITI"])
        employment = st.selectbox("Employment", ["", "Student", "Unemployed", "Employed", "Business", "Farmer", "Retired"])
        income = st.selectbox("Income", ["", "Below ₹1L", "₹1-2.5L", "₹2.5-5L", "Above ₹5L"])
        all_states = [
            "", "Tamil Nadu", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", 
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", 
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", 
            "Delhi", "Jammu & Kashmir", "Ladakh", "Puducherry"
        ]
        state = st.selectbox("State", all_states)
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about schemes..."):
    if collection is None:
        st.error("Database not available. Please check initialization.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching verified schemes..."):
            try:
                user_state = st.session_state.profile.get('state', '').strip()
                user_edu = st.session_state.profile.get('education', '').strip()
                
                # Context query focused on intent + education
                query_tokens = [prompt]
                if user_edu:
                    query_tokens.append(f"{user_edu} college education scholarship")
                if user_state:
                    query_tokens.append(user_state)
                search_query = " ".join(query_tokens)
                
                query_vector = embed_model.encode(search_query).tolist()
                
                results = collection.query(
                    query_embeddings=[query_vector], 
                    n_results=25,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results['documents'] and results['documents'][0]:
                    filtered_docs = []
                    filtered_metas = []
                    all_india_docs = []
                    
                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        doc_lower = doc.lower()
                        
                        is_all_india = any(term in doc_lower for term in [
                            'all india', 'central', 'national', 'all states', 
                            'ministry of', 'government of india'
                        ])
                        
                        if user_state and user_state.lower() in doc_lower:
                            filtered_docs.append(doc)
                            filtered_metas.append(metadata)
                        elif is_all_india:
                            all_india_docs.append((doc, metadata))
                    
                    for doc, metadata in all_india_docs[:3]:
                        filtered_docs.append(doc)
                        filtered_metas.append(metadata)
                    
                    filtered_docs = filtered_docs[:6]
                    filtered_metas = filtered_metas[:6]

                    if filtered_docs:
                        enhanced_prompt = get_enhanced_prompt(
                            query=prompt,
                            context_chunks=filtered_docs,
                            user_profile=st.session_state.profile,
                            user_state=user_state
                        )
                        
                        response = gemini_model.generate_content(enhanced_prompt)
                        response_text = response.text
                        
                        scheme_names = [m.get('scheme_name', 'Scheme') for m in filtered_metas if m.get('scheme_name')]
                        if scheme_names:
                            unique_names = list(dict.fromkeys(scheme_names))
                            response_text += f"\n\n---\n**📌 Schemes found:** {', '.join(unique_names)}"
                        
                        st.markdown(response_text)
                    else:
                        response_text = f"No active schemes found matching this criteria for {user_state}."
                        st.info(response_text)
                else:
                    response_text = "I couldn't find any schemes. Try different keywords."
                    st.warning(response_text)
                    
            except Exception as e:
                response_text = f"Search error: {str(e)[:100]}"
                st.error(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})