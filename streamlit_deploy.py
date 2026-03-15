# streamlit_deploy.py
import os
import subprocess
import sys
import time

print("=" * 50)
print("🚀 PolicyNav Deployment Starting")
print("=" * 50)

# Check if database exists and has data
db_exists = os.path.exists("./policynav_db")

if not db_exists:
    print("\n📦 First-time setup: Creating database from CSV...")
    print("   This will take 2-3 minutes...\n")
    
    # Run database initialization
    result = subprocess.run([sys.executable, "init_db.py"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print("\n✅ Database initialization complete!")
else:
    print("\n📦 Database already exists, skipping initialization")
    
    # Verify database has data
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./policynav_db")
        collection = client.get_collection("tamilnadu_schemes")
        count = collection.count()
        print(f"   • Found {count} chunks in existing database")
    except:
        print("   • Database empty or corrupted, will recreate")
        os.system(f"rm -rf ./policynav_db")
        subprocess.run([sys.executable, "init_db.py"])

print("\n🚀 Starting PolicyNav app...")
print("=" * 50 + "\n")

# Run the main app
os.system("streamlit run app.py")