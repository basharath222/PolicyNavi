# load_complete_data.py
import json
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
import os

print("🚀 Loading complete scheme data into database...")

# Load schemes
with open('tn_schemes_complete.json', 'r', encoding='utf-8') as f:
    schemes = json.load(f)

print(f"📊 Loaded {len(schemes)} schemes with details")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./policynav_db")
try:
    client.delete_collection("tamilnadu_schemes")
    print("🗑️ Deleted old collection")
except:
    pass

collection = client.create_collection(name="tamilnadu_schemes")
print("✨ Created new collection")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Process each scheme
total_chunks = 0

for idx, scheme in enumerate(schemes):
    print(f"\n[{idx+1}/{len(schemes)}] Processing: {scheme['name']}")
    
    # Create multiple chunks per scheme
    chunks = [
        f"Scheme Name: {scheme['name']}\nCategory: {scheme['category']}\nState: Tamil Nadu",
        f"Details about {scheme['name']}: {scheme['details'][:500]}",
        f"How to apply for {scheme['name']}: Visit official website or District Industries Centre. More info: {scheme['url']}",
        f"Eligibility for {scheme['name']}: Check official website for detailed criteria. Category: {scheme['category']}"
    ]
    
    # Add chunks to database
    for chunk in chunks:
        if len(chunk) > 50:  # Only add meaningful chunks
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "scheme_name": scheme['name'],
                    "category": scheme['category'],
                    "state": "Tamil Nadu",
                    "url": scheme['url']
                }]
            )
            total_chunks += 1
    
    print(f"  ✓ Added {len(chunks)} chunks")

print(f"\n✅ Database ready!")
print(f"   • {len(schemes)} schemes")
print(f"   • {total_chunks} total chunks")
print(f"   • Database location: ./policynav_db")

# Show sample
sample = collection.peek()
if sample and sample['documents']:
    print("\n📝 Sample data:")
    for i, doc in enumerate(sample['documents'][:2]):
        print(f"\nChunk {i+1}: {doc[:100]}...")