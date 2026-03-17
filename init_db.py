# init_db.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os

print("🚀 Initializing database for deployment...")
print("=" * 50)

# Check if CSV exists
csv_path = 'cleaned_my_scheme_data_fixed.csv'
if not os.path.exists(csv_path):
    print(f"❌ CSV file not found at {csv_path}")
    exit(1)

# Load CSV data
print(f"📂 Loading CSV file...")
df = pd.read_csv(csv_path)
print(f"📊 Loaded {len(df)} total schemes")

# Initialize ChromaDB
print(f"\n🔄 Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./policynav_db")

# Delete old collection if exists
try:
    client.delete_collection("indian_schemes")
    print("🗑️ Deleted old collection")
except:
    print("✨ Creating new collection")

# Create new collection
collection = client.create_collection(name="indian_schemes")

# Load embedding model
print(f"\n🤖 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✅ Model loaded successfully")

# Process all schemes (NO FILTERING)
print(f"\n📥 Adding {len(df)} schemes to database...")
print("-" * 50)

success_count = 0
error_count = 0

for idx, row in df.iterrows():
    try:
        # Create rich text from all columns
        text_parts = []
        scheme_metadata = {}
        
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                value = str(row[col]).strip()
                text_parts.append(f"{col}: {value}")
                
                # Store important fields in metadata for filtering
                if col.lower() in ['scheme_name', 'name', 'title']:
                    scheme_metadata['scheme_name'] = value[:100]
                elif col.lower() in ['state', 'states', 'beneficiary_states']:
                    scheme_metadata['state'] = value
                elif col.lower() in ['category', 'beneficiary_category']:
                    scheme_metadata['category'] = value
                elif col.lower() in ['ministry', 'department']:
                    scheme_metadata['ministry'] = value
        
        # Combine all text
        text = "\n".join(text_parts)
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Ensure scheme_name exists in metadata
        if 'scheme_name' not in scheme_metadata:
            scheme_metadata['scheme_name'] = f"Scheme_{idx}"
        
        # Add state info if available
        if 'state' not in scheme_metadata:
            # Try to detect state from text
            text_lower = text.lower()
            if 'tamil nadu' in text_lower or 'tn ' in text_lower:
                scheme_metadata['state'] = 'Tamil Nadu'
            elif 'kerala' in text_lower:
                scheme_metadata['state'] = 'Kerala'
            elif 'karnataka' in text_lower:
                scheme_metadata['state'] = 'Karnataka'
            elif 'maharashtra' in text_lower:
                scheme_metadata['state'] = 'Maharashtra'
            elif 'delhi' in text_lower:
                scheme_metadata['state'] = 'Delhi'
            elif 'all india' in text_lower or 'central' in text_lower:
                scheme_metadata['state'] = 'All-India'
            else:
                scheme_metadata['state'] = 'Unknown'
        
        # Add to database
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[scheme_metadata]
        )
        success_count += 1
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  ✅ Processed {idx + 1}/{len(df)} schemes...")
            
    except Exception as e:
        error_count += 1
        print(f"  ❌ Error on row {idx}: {str(e)[:50]}")
        continue

print("-" * 50)
print(f"\n✅ Database creation complete!")
print(f"   • Successfully loaded: {success_count} schemes")
print(f"   • Errors: {error_count}")
print(f"   • Total in database: {collection.count()} chunks")
print(f"   • Database location: ./policynav_db")

# Show sample of states in database
print(f"\n📊 Sample of states in database:")
try:
    results = collection.peek()
    states_seen = set()
    for meta in results['metadatas'][:10]:
        if meta.get('state'):
            states_seen.add(meta['state'])
    print(f"   • States found: {', '.join(list(states_seen)[:5])}")
except:
    pass

print("\n🚀 You can now run the app with: streamlit run app.py")
# At the end of init_db.py, add verification
try:
    # Verify the database was created
    client = chromadb.PersistentClient(path="./policynav_db")
    collection = client.get_collection("indian_schemes")
    final_count = collection.count()
    print(f"\n✅ FINAL VERIFICATION: {final_count} chunks in database")
    print(f"📁 Database location: {os.path.abspath('./policynav_db')}")
    
    # List a few sample schemes
    if final_count > 0:
        sample = collection.peek()
        print("\n📝 Sample schemes:")
        for i, meta in enumerate(sample['metadatas'][:3]):
            print(f"  {i+1}. {meta.get('scheme_name', 'Unknown')}")
except Exception as e:
    print(f"⚠️ Final verification failed: {e}")