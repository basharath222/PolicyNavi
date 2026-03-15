# init_db.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
import streamlit as st

print("🚀 Initializing database for deployment...")

# Check if CSV exists
csv_path = 'cleaned_my_scheme_data_fixed.csv'
if not os.path.exists(csv_path):
    st.error(f"❌ CSV file not found at {csv_path}")
    st.stop()

# Load your CSV data
df = pd.read_csv(csv_path)
print(f"📊 Loaded {len(df)} total schemes")

# Filter for Tamil Nadu
if 'state' in df.columns:
    df_tn = df[df['state'].str.contains('Tamil Nadu|TN|TAMIL NADU', case=False, na=False)]
else:
    # If no state column, try to infer from scheme names
    tn_keywords = ['tamil nadu', 'chennai', 'coimbatore', 'madurai', 'trichy', 'salem']
    mask = df.apply(lambda row: any(kw in str(row).lower() for kw in tn_keywords), axis=1)
    df_tn = df[mask]

print(f"📊 Found {len(df_tn)} Tamil Nadu schemes")

if len(df_tn) == 0:
    st.error("❌ No Tamil Nadu schemes found in CSV!")
    st.stop()

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./policynav_db")
try:
    client.delete_collection("tamilnadu_schemes")
    print("🗑️ Deleted old collection")
except:
    print("✨ Creating new collection")

collection = client.create_collection(name="tamilnadu_schemes")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add schemes to database
success_count = 0
for idx, row in df_tn.iterrows():
    try:
        # Create rich text from all columns
        text_parts = []
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(f"{col}: {row[col]}")
        
        text = "\n".join(text_parts)
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Get scheme name
        scheme_name = row.get('scheme_name', row.get('name', row.get('title', f'Scheme_{idx}')))
        
        # Add to database
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "scheme_name": str(scheme_name)[:100],
                "state": "Tamil Nadu",
                "index": idx
            }]
        )
        success_count += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  ✅ Added {idx + 1} schemes...")
            
    except Exception as e:
        print(f"  ❌ Error on row {idx}: {e}")
        continue

print(f"\n✅ Database ready!")
print(f"   • {success_count} schemes successfully loaded")
print(f"   • Database location: ./policynav_db")

# Verify
try:
    count = collection.count()
    print(f"   • Total chunks: {count}")
except:
    pass