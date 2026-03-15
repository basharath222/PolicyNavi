import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(URL, KEY)

# --- 1. CHAT SESSION (THREAD) LOGIC ---

def create_chat_session(user_id, title="New Conversation"):
    """Creates a new unique thread for a user and returns the session_id."""
    try:
        # Ensure we're using the authenticated client
        if not supabase.auth.get_session():
            raise Exception("Not authenticated")
            
        res = supabase.table("chat_sessions").insert({
            "user_id": user_id,
            "title": title
        }).execute()
        return res.data[0]['id'] if res.data else None
    except Exception as e:
        print(f"Error creating chat session: {e}")
        return None

def get_user_sessions(user_id):
    """Fetches all past chat threads for the sidebar list."""
    try:
        if not supabase.auth.get_session():
            return []
            
        res = (
            supabase.table("chat_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        print(f"Error getting user sessions: {e}")
        return []

# --- 2. CHAT HISTORY LOGIC ---

def save_chat_message(session_id, role, content):
    """Stores a single message linked to a specific session."""
    try:
        # First verify the session belongs to the current user
        if not supabase.auth.get_session():
            raise Exception("Not authenticated")
            
        # Get current user
        user = supabase.auth.get_user()
        if not user:
            raise Exception("No user found")
            
        # Verify session ownership
        session_check = (
            supabase.table("chat_sessions")
            .select("user_id")
            .eq("id", session_id)
            .eq("user_id", user.user.id)
            .execute()
        )
        
        if not session_check.data:
            raise Exception("Session not found or doesn't belong to user")
            
        # Insert the message
        res = supabase.table("chat_history").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
        
        return res.data if res.data else None
    except Exception as e:
        print(f"Error saving chat message: {e}")
        raise e

def update_chat_title(session_id, title):
    """Updates the 'New Conversation' text to something meaningful."""
    try:
        if not supabase.auth.get_session():
            return
            
        supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()
    except Exception as e:
        print(f"Error updating chat title: {e}")

def get_chat_history(session_id, limit=50):
    """Strictly fetches history ONLY for this specific thread."""
    try:
        if not supabase.auth.get_session():
            return []
            
        res = (
            supabase.table("chat_history")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data if res.data else []
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []

# --- 3. USER PROFILE LOGIC ---

def save_user_profile(user_id, profile_dict):
    """Saves detected traits (age, gender, etc) to the cloud."""
    try:
        if not supabase.auth.get_session():
            return
            
        payload = {
            "user_id": user_id,
            "data": profile_dict,
            "updated_at": "now()"
        }
        supabase.table("user_profiles").upsert(payload).execute()
    except Exception as e:
        print(f"Error saving user profile: {e}")

def get_user_profile(user_id):
    """Retrieves the flexible data blob for AI context."""
    try:
        if not supabase.auth.get_session():
            return {}
            
        res = supabase.table("user_profiles").select("data").eq("user_id", user_id).execute()
        return res.data[0]['data'] if res.data else {}
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return {}

# --- 4. SESSION METADATA (Optional) ---

def update_session_metadata(session_id, metadata):
    """Update session metadata like filters used"""
    try:
        if not supabase.auth.get_session():
            return
            
        supabase.table("chat_sessions").update({"metadata": metadata}).eq("id", session_id).execute()
    except Exception as e:
        print(f"Error updating session metadata: {e}")

# --- 5. SCHEME FEEDBACK (Optional) ---

def save_scheme_feedback(session_id, scheme_name, feedback, notes=None):
    """Save user feedback on schemes"""
    try:
        if not supabase.auth.get_session():
            return
            
        user = supabase.auth.get_user()
        if not user:
            return
            
        res = supabase.table("scheme_feedback").insert({
            "user_id": user.user.id,
            "session_id": session_id,
            "scheme_name": scheme_name,
            "feedback": feedback,
            "notes": notes
        }).execute()
        return res.data
    except Exception as e:
        print(f"Error saving scheme feedback: {e}")
        return None