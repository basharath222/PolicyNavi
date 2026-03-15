# load_to_chromadb.py
import json
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import time
import random

print("🚀 Loading schemes into database...")

# Load scheme list
with open('tn_schemes_list.json', 'r') as f:
    schemes = json.load(f)

print(f"📊 Loaded {len(schemes)} schemes from JSON")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./policynav_db")
try:
    client.delete_collection("tamilnadu_schemes")
    print("🗑️ Deleted old collection")
except:
    pass

collection = client.create_collection(name="tamilnadu_schemes")
print("✨ Created new collection")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each scheme
total_chunks = 0

for idx, scheme in enumerate(schemes[:100]):  # Limit to 100 for now
    print(f"\n[{idx+1}/{min(100, len(schemes))}] Processing: {scheme['name']}")
    
    # Create chunks from scheme name and basic info
    chunks = [
        f"Scheme Name: {scheme['name']}",
        f"This scheme is available in Tamil Nadu. For more details, visit: {scheme['url']}",
        f"To apply for {scheme['name']}, visit your nearest District Industries Centre or check the official website."
    ]
    
    # Try to fetch basic info (optional - can be slow)
    try:
        response = requests.get(scheme['url'], timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()[:1000]  # First 1000 chars
            chunks.append(f"Scheme Information: {text}")
    except:
        pass
    
    # Add to database
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "scheme_name": scheme['name'],
                "state": "Tamil Nadu",
                "url": scheme['url']
            }]
        )
        total_chunks += 1
    
    print(f"  ✓ Added {len(chunks)} chunks")
    time.sleep(random.uniform(1, 2))  # Be gentle to the server

print(f"\n✅ Database ready! {total_chunks} chunks for {min(100, len(schemes))} schemes")
print(f"📁 Database location: ./policynav_db")