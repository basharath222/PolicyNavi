# init_db.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys

print("🚀 Initializing database for PolicyNav...")
print("=" * 60)

csv_path = 'cleaned_my_scheme_data_fixed.csv'
db_path = './policynav_db'
collection_name = 'indian_schemes'

if not os.path.exists(csv_path):
    print(f"❌ Error: CSV file not found at {csv_path}")
    sys.exit(1)

# 1. Load Dataset
print(f"📂 Loading {csv_path}...")
df = pd.read_csv(csv_path).fillna("")
print(f"📊 Loaded {len(df)} total schemes.")

# 2. Setup ChromaDB Local Storage
print(f"\n🔄 Connecting to ChromaDB storage at {db_path}...")
client = chromadb.PersistentClient(path=db_path)

try:
    client.delete_collection(name=collection_name)
    print("🗑️ Removed existing collection to ensure a fresh build.")
except Exception:
    pass

collection = client.create_collection(name=collection_name)

# 3. Load Embedding Model
print("\n🤖 Loading Sentence-Transformer ('all-MiniLM-L6-v2')...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully.")

# 4. Prepare Chunks and Metadata
print("\n📦 Structuring scheme records for indexing...")
documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    # Build a complete readable document block
    text_parts = []
    for col in df.columns:
        val = str(row[col]).strip()
        if val:
            text_parts.append(f"{col}: {val}")
    doc_chunk = "\n".join(text_parts)

    # Extract metadata fields for state/demographic guardrails
    scheme_name = str(row.get('scheme_name', row.get('name', f'Scheme_{idx}'))).strip()
    state = str(row.get('state', row.get('states', 'All-India'))).strip()
    category = str(row.get('category', row.get('beneficiary_category', 'General'))).strip()
    url = str(row.get('url', row.get('source_url', row.get('scheme_link', '')))).strip()

    # Fallback state detection if missing
    if not state or state.lower() in ['unknown', 'nan', '']:
        text_lower = doc_chunk.lower()
        if 'tamil nadu' in text_lower or 'tn ' in text_lower:
            state = 'Tamil Nadu'
        elif 'karnataka' in text_lower:
            state = 'Karnataka'
        elif 'kerala' in text_lower:
            state = 'Kerala'
        elif any(k in text_lower for k in ['all india', 'central', 'national']):
            state = 'All-India'
        else:
            state = 'All-India'

    documents.append(doc_chunk)
    metadatas.append({
        "scheme_name": scheme_name[:100],
        "state": state,
        "category": category[:50],
        "url": url
    })
    ids.append(f"scheme_{idx}")

# 5. Batch Encoding (Crucial: 20x faster than row-by-row)
print(f"\n⚡ Batch encoding and indexing {len(documents)} schemes...")
batch_size = 128
total_batches = (len(documents) + batch_size - 1) // batch_size

for b_idx in range(total_batches):
    start = b_idx * batch_size
    end = min(start + batch_size, len(documents))
    
    b_docs = documents[start:end]
    b_metas = metadatas[start:end]
    b_ids = ids[start:end]

    # Vectorize the entire batch together
    b_embeddings = model.encode(b_docs, batch_size=batch_size, show_progress_bar=False).tolist()

    collection.add(
        ids=b_ids,
        embeddings=b_embeddings,
        documents=b_docs,
        metadatas=b_metas
    )
    print(f"   Indexed batch {b_idx + 1}/{total_batches} ({end}/{len(documents)} records)")

print("=" * 60)
print(f"✅ Success! Vector database created with {collection.count()} chunks.")
print(f"📁 Local storage directory: {os.path.abspath(db_path)}")