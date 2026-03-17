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
CSV_PATH = "cleaned_my_scheme_data_fixed.csv"

# --- Windows event loop fix ---
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'profile' not in st.session_state:
    st.session_state.profile = {}
# --- AUTO-INITIALIZE DATABASE IF EMPTY ---
def check_and_init_database():
    """Check if database has data, if not run init_db.py"""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        embed_model, gemini_model, collection = load_models()

        try:
            count = collection.count()
            st.sidebar.success(f"✅ {count} schemes loaded")
        except Exception as e:
            st.sidebar.error(f"Database error: {e}")
            # Try to reconnect
            try:
                db_client = chromadb.PersistentClient(path=DB_PATH)
                collection = db_client.get_collection(name="indian_schemes")
                count = collection.count()
                st.sidebar.success(f"✅ {count} schemes loaded (reconnected)")
            except:
                st.sidebar.error("Database not found")
        
        # Database doesn't exist or is empty
        st.warning("📦 Database is empty. Initializing with 3000+ schemes...")
        
        import subprocess
        result = subprocess.run([sys.executable, "init_db.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("✅ Database initialized! Refreshing...")
            st.rerun()
        else:
            st.error(f"Database init failed: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Database check failed: {e}")
        return False

# --- DATABASE INITIALIZATION FOR DEPLOYMENT ---
import subprocess
import time

# Function to check if database has data
def check_database():
    try:
        # Try to connect to ChromaDB
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            collection = client.get_collection(name="indian_schemes")
            count = collection.count()
            return count > 0, count
        except:
            # Collection doesn't exist
            return False, 0
    except Exception as e:
        print(f"Database check error: {e}")
        return False, 0

# Check database status
db_ready, scheme_count = check_database()

if not db_ready:
    st.warning("""
    ### 📦 First-time Setup Required
    
    The database is being initialized with 3000+ schemes. This happens only once during deployment.
    Please wait 2-3 minutes...
    """)
    
    # Create a placeholder for logs
    log_placeholder = st.empty()
    
    try:
        with st.spinner("🔄 Initializing database... This may take a few minutes..."):
            # Run init_db.py and capture output in real-time
            process = subprocess.Popen(
                [sys.executable, "init_db.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Show output in real-time
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                # Show last 5 lines
                with log_placeholder.expander("📋 Setup Progress", expanded=True):
                    st.code("".join(output_lines[-10:]))
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                st.success("✅ Database initialized successfully!")
                # Check again
                db_ready, scheme_count = check_database()
                if db_ready:
                    st.success(f"✅ {scheme_count} schemes loaded!")
                    st.rerun()
                else:
                    st.error("Database initialization completed but no schemes found!")
            else:
                # Show error
                error_output = process.stderr.read()
                st.error(f"Database init failed: {error_output[:500]}")
                
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
else:
    st.sidebar.success(f"✅ {scheme_count} schemes loaded")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in secrets!")
        st.stop()
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    
    db_client = chromadb.PersistentClient(path=DB_PATH)
    db_collection = db_client.get_or_create_collection(name="indian_schemes")
    return embed_model, llm_model, db_collection

embed_model, gemini_model, collection = load_models()

# --- ENHANCED PROMPT WITH STATE FILTERING (FIXED) ---

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

## 📚 RETRIEVED SCHEMES (CONTEXT ONLY - USE THESE)
{formatted_context}

## 🎯 RESPONSE INSTRUCTIONS - FOLLOW EXACTLY

### 1. STATE FILTERING (CRITICAL)
- User is from: **{user_state if user_state else 'Unknown'}**
- **ONLY** recommend schemes that are:
  a) Specifically for {user_state}
  b) "All-India" / "Central" / "National" schemes
  c) If a scheme mentions other states (Kerala, Karnataka, etc.), **DO NOT** recommend it to this user

### 2. INCLUSIVE MATCHING - CONNECT THE DOTS
- **"Post-Matric"** = includes 12th pass, graduation, and ALL higher education
- **"Scholarship"** = financial aid for students (any level)
- **"SC/ST/OBC"** = includes the specific category the user belongs to
- **"Professional courses" / "Degree courses" / "Technical Education"** = includes engineering
- **"Diploma"** = includes polytechnic/engineering diplomas
- **"Post-Matric Scholarship"** in ANY context = applies to students after 10th (11th, 12th, graduation)
- If a scheme name contains **"Scholarship"** and the user is a student, it is HIGHLY RELEVANT

### 3. FOR SCHOLARSHIP QUERIES (SPECIAL HANDLING)
When user asks about scholarships for SC students:
- Look for ANY scheme containing: "scholarship", "Post-Matric", "SC", "ST", "Postmatric"
- "Postmatric Scholarship" in the context ALWAYS applies to 12th pass students
- Do NOT exclude a scheme just because it doesn't explicitly say "engineering" - if it's a scholarship for higher education, it can be used for engineering
- If a scheme is for "BC/MBC" but also mentions "SC" anywhere in the text, include it
- If a scheme is for "Minorities" but the user is SC, DO NOT include it

### 4. RESPONSE STRUCTURE

#### For EACH relevant scheme, provide:
### 🏷️ Scheme Name
**📝 Details:** (2-3 sentences about the scheme)

**✅ Eligibility:** (Bullet points explaining who can apply)
- ✓ **Matches your profile:** [explain what matches]
- ⚠️ **Note:** [explain any differences or requirements]

**💰 Benefits:** (What the user gets - amounts, facilities, etc.)

**📄 Application Process:** (Step-by-step instructions)

**📑 Documents Required:** (Bullet list)

**🔗 Source URL:** (If available in context)

#### If MULTIPLE schemes apply:
- List each scheme with full details as above
- Add a comparison table at the end:
  | Scheme Name | Target Group | Key Benefit | State |

#### If PARTIAL matches exist:
- Include them with clear explanations
- Example: "This scheme is for BC/MBC students, but also mentions SC eligibility. You should verify with the department."

#### If NO schemes match:
- Check if there are ANY scholarship schemes in the context
- If there are, explain why they don't match and suggest alternatives
- If none, say: "I couldn't find any schemes matching your exact criteria. Would you like to try a different search?"

### 5. EXAMPLE FOR SCHOLARSHIP QUERY

### 🏷️ Post-Matric Scholarship for SC/ST Students
**📝 Details:** A scholarship for Scheduled Caste and Scheduled Tribe students pursuing education after Class 10, including engineering degrees.

**✅ Eligibility:**
- ✓ You belong to SC category (matches your profile)
- ✓ You have passed 12th and are pursuing higher education
- ✓ You are from {user_state}
- Family income should not exceed ₹2.5 lakh per annum
- Must be enrolled in a recognized institution

**💰 Benefits:**
- Full tuition fee reimbursement
- Monthly maintenance allowance
- Book grant

**📄 Application Process:**
1. Apply through National Scholarship Portal
2. Register and fill application
3. Upload documents
4. Submit to institution for verification

**📑 Documents Required:**
- Aadhaar Card
- Community Certificate
- Marksheets
- Income Certificate
- Bank Details

**🔗 Source URL:** scholarships.gov.in

### 6. CRITICAL RULES
- **NEVER** invent information not in the context
- **ALWAYS** explain why a scheme applies or doesn't apply
- If details are missing, say what's available
- Be helpful, clear, and solution-oriented
- Do NOT use apologetic language ("I'm sorry", "Unfortunately")

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
    try:
        count = collection.count()
        st.success(f"✅ {count} schemes loaded")
    except:
        st.error("Database not found")
    
    st.divider()
    
    # User profile - ADDED STATE FIELD
    with st.expander("📝 Your Profile", expanded=True):
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        education = st.selectbox("Education", ["", "10th", "12th", "Graduate", "Diploma", "ITI"])
        employment = st.selectbox("Employment", ["", "Unemployed", "Employed", "Student", "Business", "Farmer", "Retired"])
        income = st.selectbox("Income", ["", "Below ₹1L", "₹1-2.5L", "₹2.5-5L", "Above ₹5L"])
        
        # CRITICAL: Add state field
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
                "state": state,  # Save state
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
# --- MAIN CHAT INTERFACE ---
st.title("📜 PolicyNav - Indian Government Scheme Advisor")
st.caption("Ask about schemes • I'll find what you're eligible for")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about schemes..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching schemes..."):
            
            # Get user state from profile
            user_state = st.session_state.profile.get('state', '')
            
            # Create search query with state context
            profile_text = " ".join([str(v) for v in st.session_state.profile.values() if v])
            search_query = f"{user_state} {prompt} {profile_text}"
            
            try:
                # Search database - with error handling
                query_vector = embed_model.encode(search_query).tolist()
                
                # IMPORTANT: Refresh collection reference before query
                db_client = chromadb.PersistentClient(path=DB_PATH)
                collection = db_client.get_collection(name="indian_schemes")
                
                results = collection.query(
                    query_embeddings=[query_vector], 
                    n_results=15,
                    include=["documents", "metadatas", "distances"]
                )

                if results and results['documents'] and results['documents'][0]:
                    # Filter results by state and relevance
                    filtered_docs = []
                    filtered_metas = []
                    seen_schemes = set()

                    for i, doc in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i]
                        doc_lower = doc.lower()
                        scheme_name = metadata.get('scheme_name', '')
                        
                        if scheme_name in seen_schemes:
                            continue
                        
                        distance = results['distances'][0][i] if results['distances'] else 0
                        relevance = 1 - distance
                        
                        if user_state:
                            state_match = (
                                user_state.lower() in doc_lower or
                                'all india' in doc_lower or
                                'central' in doc_lower or
                                'national' in doc_lower or
                                'all states' in doc_lower
                            )
                            
                            metadata_state = str(metadata.get('state', '')).lower()
                            metadata_match = (
                                user_state.lower() in metadata_state or
                                'all india' in metadata_state
                            )
                            
                            is_relevant = state_match or metadata_match or relevance > 0.6
                            
                            if is_relevant:
                                filtered_docs.append(doc)
                                filtered_metas.append(metadata)
                                seen_schemes.add(scheme_name)
                        else:
                            if relevance > 0.5:
                                filtered_docs.append(doc)
                                filtered_metas.append(metadata)
                                seen_schemes.add(scheme_name)
                        
                        if len(filtered_docs) >= 7:
                            break

                    if filtered_docs:
                        context_for_prompt = "\n\n---\n\n".join(filtered_docs)
                        
                        profile_lines = []
                        for k, v in st.session_state.profile.items():
                            if v and str(v).strip():
                                profile_lines.append(f"{k}: {v}")
                        profile_text = " | ".join(profile_lines) if profile_lines else "No profile provided"
                        
                        enhanced_prompt = get_enhanced_prompt(
                            query=prompt,
                            context_chunks=filtered_docs,
                            user_profile=st.session_state.profile,
                            user_state=user_state
                        )
                        
                        messages = [{"role": "user", "parts": [enhanced_prompt]}]
                        response = gemini_model.generate_content(messages)
                        response_text = response.text
                        
                        scheme_names = [m.get('scheme_name', 'Scheme') for m in filtered_metas if m.get('scheme_name')]
                        if scheme_names:
                            response_text += f"\n\n---\n**📌 Schemes found:** {', '.join(scheme_names)}"
                        
                        st.markdown(response_text)
                    else:
                        st.warning(f"No specific schemes found for {user_state}. Showing general results:")
                        for i, doc in enumerate(results['documents'][0][:3]):
                            scheme_name = results['metadatas'][0][i].get('scheme_name', 'Scheme')
                            st.markdown(f"- **{scheme_name}**")
                        
                        response_text = f"No specific schemes found for {user_state}. Try a different search."
                else:
                    response_text = "I couldn't find any schemes. Try different keywords."
                    st.warning(response_text)
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"Database query error: {error_msg[:100]}")
                response_text = "Sorry, I encountered an error while searching. Please try again."
            
            # Save response
            st.session_state.messages.append({"role": "assistant", "content": response_text})