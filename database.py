import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Initialize Supabase Connection
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in .env")

# Create the cloud client
supabase: Client = create_client(URL, KEY)

def init_db():
    """
    Ensure tables are created in Supabase SQL Editor first.
    """
    pass

# --- USER PROFILE LOGIC ---

def save_user_profile(user_id, profile_dict):
    """Saves a flexible dictionary of user traits (gender, age, etc)."""
    payload = {
        "user_id": user_id,
        "data": profile_dict
    }
    # upsert handles both insert and update automatically
    supabase.table("user_profiles").upsert(payload).execute()

def get_user_profile(user_id):
    """Retrieves the flexible data blob from the cloud."""
    try:
        response = supabase.table("user_profiles").select("data").eq("user_id", user_id).execute()
        # Return the 'data' field if found, otherwise return an empty dict
        return response.data[0]['data'] if response.data else {}
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return {}

# --- CHAT HISTORY LOGIC ---

def save_chat_message(user_id, role, content):
    """Stores a single message in the chat_history table."""
    data = {
        "user_id": user_id,
        "role": role,
        "content": content
    }
    supabase.table("chat_history").insert(data).execute()

def get_chat_history(user_id, limit=10):
    """Fetches the last N messages for the user from the cloud."""
    try:
        response = (
            supabase.table("chat_history")
            .select("role, content")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        # Reverse to return messages in chronological order (oldest to newest)
        return [{"role": r['role'], "content": r['content']} for r in reversed(response.data)]
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []