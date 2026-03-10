import sqlite3

def init_db():
    conn = sqlite3.connect('policynav_users.db')
    c = conn.cursor()
    # Table for User Profiles
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, location TEXT, category TEXT, business TEXT)''')
    # Table for Chat History
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (username TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_user_profile(username, location, category, business):
    conn = sqlite3.connect('policynav_users.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)", (username, location, category, business))
    conn.commit()
    conn.close()