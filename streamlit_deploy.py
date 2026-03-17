# streamlit_deploy.py
import os
import subprocess
import sys
import time

print("=" * 50)
print("🚀 PolicyNav Deployment Starting")
print("=" * 50)

# Check if database exists and has data
db_path = "./policynav_db"
db_ready = False

if os.path.exists(db_path):
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        if len(collections) > 0:
            collection = client.get_collection("indian_schemes")
            count = collection.count()
            if count > 0:
                print(f"✅ Database already exists with {count} schemes")
                db_ready = True
    except Exception as e:
        print(f"⚠️ Database check failed: {e}")

if not db_ready:
    print("\n📦 First-time setup: Creating database from CSV...")
    print("   This will take 2-3 minutes...\n")
    
    # Run database initialization
    result = subprocess.run([sys.executable, "init_db.py"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode == 0:
        print("\n✅ Database initialization complete!")
    else:
        print("\n❌ Database initialization failed!")

print("\n🚀 Starting PolicyNav app...")
print("=" * 50 + "\n")

# Run the main app
os.system("streamlit run app.py")