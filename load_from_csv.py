# load_from_csv.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# Load the CSV
df = pd.read_csv('cleaned_my_scheme_data_fixed.csv')

# Filter for Tamil Nadu if there's a state column
if 'state' in df.columns:
    df_tn = df[df['state'].str.contains('Tamil Nadu|TN|TAMIL NADU', case=False, na=False)]
else:
    df_tn = df  # Use all if no state column

print(f"Found {len(df_tn)} Tamil Nadu schemes")

# Load into ChromaDB
client = chromadb.PersistentClient(path="./policynav_db")
try:
    client.delete_collection("tamilnadu_schemes")
except:
    pass

collection = client.create_collection(name="tamilnadu_schemes")
model = SentenceTransformer("all-MiniLM-L6-v2")

for _, row in df_tn.iterrows():
    # Create text from all columns
    text = ' '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
    
    embedding = model.encode(text).tolist()
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "scheme_name": row.get('scheme_name', row.get('name', 'Unknown')),
            "state": "Tamil Nadu"
        }]
    )

print(f"✅ Loaded into database")