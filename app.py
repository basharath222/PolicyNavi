import streamlit as st
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PolicyNav - Tamil Nadu Schemes", 
    page_icon="🎯",
    layout="wide"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    
    db_client = chromadb.PersistentClient(path="./policynav_db")
    db_collection = db_client.get_or_create_collection(name="tamilnadu_schemes")
    return embed_model, llm_model, db_collection

embed_model, gemini_model, collection = load_models()

# --- SYSTEM PROMPT (from your reference) ---
SYSTEM_PROMPT = """You are an expert assistant on Indian government schemes. You have very detailed knowledge about government schemes. Your task is to find and present scheme details from the provided context only—no external data or guesses. Your response should be perfectly aligned with the user’s query.  

## 1. Core Principles  

1. **Context-Only**  
   - Rely solely on retrieved document chunks. Do not add, infer, or hallucinate information.  
2. **User Focus**  
   - Read the user’s query in full. Address exactly what they ask, no more, no less.  
3. **Similarity Thresholds**  
   - If a chunk’s content overlaps ≥ 60% with the query (keywords, entities, intent), treat it as relevant.  
   - Only consider chunks with ≥ 50% semantic relevance.   
4. **Entity Sensitivity**  
   - If the user names any entity—state, district, qualification, beneficiary group (student, woman, senior citizen, person with disability), religion, caste, etc.—treat it as a mandatory filter.  
   - Only include schemes whose metadata (e.g. target_beneficiaries_states, eligibility, tags) explicitly list that entity.  
   - Exclude any scheme missing that entity.
   - If Any scheme mention other entity than query of same type(e.g. If Maharashtra is mention in query but Andhra Pradesh mention in scheme) then discard that scheme.
5. **All-India Schemes**  
   - Nationwide schemes may be included only if no more specific regional scheme exists.  
   - Clearly label such schemes as “All-India.”    
6. **Prompt Adherence**  
   - Follow every instruction exactly. Do not omit or reorder rules.  

7. Based on the retrieved information, list schemes that explicitly match all the specific criteria provided in the query, such as gender, religion, category, caste, location, or any other attribute mentioned. Only include schemes where the target beneficiaries align with each of the specified attributes, and exclude any schemes that do not meet all the mentioned criteria.

## 2. How to Handle Queries  

### A. Direct or Close Match  
- If a chunk directly or nearly answers the query, extract and display under these headers:  
  ### Scheme Name  
  ### Details  
  ### Eligibility  
  ### Benefits  
  ### Application Process  
  ### Documents Required  
  ### Source URL  

- Choose the two chunks with the highest relevance scores.  

### B. Partial or Near-Match  
- If a chunk is similar but not exact, still present it .  
- Explicitly note any mismatch or missing criteria (e.g., “This scheme applies to Gujarat, not Maharashtra”).  

### C. Multiple Schemes  
- If more than one scheme applies:  
  1. List each scheme with the above headers.  
  2. Provide a comparison on below criterias:  
     - Scheme Name
     - Target Group
     - Key Benefit
       

### D. No Match or Incomplete  
- **No match**:  
  > “I could not find any scheme matching your criteria in the provided documents. Would you like to refine your query?”  
- **Incomplete details**: If key sections are missing, list available sections and explicitly state which details are unavailable.  

## 3. Formatting Rules  
- Use markdown headings (## for main sections, ### for subsections).  
- Use concise bullet lists for Eligibility, Benefits, etc.  
- Cite each scheme’s Source URL   
- Maintain a formal, informative tone.  

## 4. Additional Guidelines  
- **Prioritization**: When multiple chunks match, prioritize those that exactly match user-specified entities and timeframes.  
- **Clarification Questions**: If the user’s query is ambiguous (e.g., “Which scheme for women?” without region), ask a follow-up to narrow focus.  
- **Example Scenario**: Optionally, if the context allows, include a brief illustrative beneficiary example.  
- **Error Handling**: Do not include apologetic or policy-excuse language (e.g., “I’m sorry…” or “Based on the provided context, I cannot…”).  
---"""

# --- EMAIL FUNCTION ---
def send_email(to_email, subject, body):
    """Send email using Gmail SMTP"""
    try:
        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        
        if not sender or not password:
            return False, "Email credentials not set in .env file"
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to_email
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'profile' not in st.session_state:
    st.session_state.profile = {}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 PolicyNav")
    st.caption("Tamil Nadu Scheme Advisor")
    
    # Database status
    try:
        count = collection.count()
        st.success(f"✅ {count} scheme chunks loaded")
        st.info(f"📊 {count//10} schemes available")  # Approximate
    except:
        st.error("❌ Database not found. Run load_from_csv.py first")
        st.stop()
    
    st.divider()
    
    # User profile
    with st.expander("📝 Your Profile", expanded=True):
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        education = st.selectbox("Education", ["", "10th", "12th", "Graduate", "Post Graduate", "Diploma", "ITI"])
        employment = st.selectbox("Employment", ["", "Unemployed", "Employed", "Student", "Business", "Farmer", "Retired"])
        income = st.selectbox("Annual Income", ["", "Below ₹1L", "₹1-2.5L", "₹2.5-5L", "Above ₹5L"])
        district = st.selectbox("District", ["", "Chennai", "Coimbatore", "Madurai", "Trichy", "Salem", "Tirunelveli", "Vellore", "Erode", "Others"])
        
        if st.button("💾 Save Profile", use_container_width=True):
            st.session_state.profile = {
                "age": age,
                "gender": gender,
                "education": education,
                "employment": employment,
                "income": income,
                "district": district
            }
            st.success("✅ Profile saved!")
    
    st.divider()
    
    # Email summary
    with st.expander("📧 Email Summary", expanded=False):
        email = st.text_input("Your Email")
        if st.button("📨 Send Chat Summary", use_container_width=True):
            if email and st.session_state.messages:
                # Create summary
                summary = "📜 PolicyNav Chat Summary\n\n"
                summary += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                summary += f"Profile: {st.session_state.profile}\n\n"
                summary += "Conversation:\n"
                for msg in st.session_state.messages[-10:]:
                    summary += f"\n{msg['role'].upper()}: {msg['content']}\n"
                
                success, msg = send_email(email, "Your PolicyNav Chat Summary", summary)
                if success:
                    st.success(msg)
                else:
                    st.error(f"Email failed: {msg}")
            else:
                st.warning("Enter email and have a chat first")
    
    st.divider()
    
    # New chat button
    if st.button("🆕 New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("📜 PolicyNav - Tamil Nadu Scheme Advisor")
st.caption(f"🔍 Searching across {count//10}+ Tamil Nadu government schemes • Ask naturally")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("e.g., I'm a woman in Chennai wanting to start a business..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching schemes..."):
            
            # Create search query with profile context
            profile_text = " ".join([f"{k}: {v}" for k, v in st.session_state.profile.items() if v])
            search_query = f"{prompt} {profile_text}"
            
            # Search database
            query_vector = embed_model.encode(search_query).tolist()
            results = collection.query(
                query_embeddings=[query_vector], 
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results['documents'] and results['documents'][0]:
                # Format context chunks
                context_chunks = []
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    score = results['distances'][0][i] if results['distances'] else 0
                    context_chunks.append(f"[Score: {1-score:.2f}] {doc}")
                
                context = "\n\n".join(context_chunks)
                
                # Create messages using the system prompt
                messages = [
                    {"role": "user", "parts": [f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}\n\nRetrieved Context:\n{context}"]}
                ]
                
                # Get response from Gemini
                response = gemini_model.generate_content(messages)
                response_text = response.text
                
                # Add scheme names from metadata
                scheme_names = set()
                for meta in results['metadatas'][0]:
                    if meta.get('scheme_name'):
                        scheme_names.add(meta['scheme_name'])
                
                if scheme_names:
                    response_text += f"\n\n---\n**📌 Schemes found:** {', '.join(list(scheme_names)[:5])}"
                
                st.markdown(response_text)
                
                # Ask for more info if needed
                if "unemployed" in prompt.lower() and not st.session_state.profile.get('education'):
                    st.info("💡 To give better recommendations, please tell me your **education qualification** in the profile section.")
                elif "business" in prompt.lower() and not st.session_state.profile.get('gender'):
                    st.info("💡 For business schemes, knowing your **gender** helps find women-specific schemes.")
                
            else:
                st.warning("I could not find any scheme matching your criteria. Would you like to refine your query?")
                response_text = "I could not find any scheme matching your criteria. Would you like to refine your query?"
            
            # Save response
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- FOOTER ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Schemes", f"{count//10}+")
with col2:
    st.metric("Your Chats", len(st.session_state.messages) // 2)
with col3:
    profile_status = "✅" if st.session_state.profile else "❌"
    st.metric("Profile", profile_status)
with col4:
    st.metric("Status", "🟢 Active")