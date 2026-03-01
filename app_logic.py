import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Gemini and ChromaDB
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API Key not found! Please check your .env file.")
else:
    genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-2.5-flash')

client = chromadb.PersistentClient(path="./policynav_db")
collection = client.get_or_create_collection(name="tamilnadu_schemes")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_policy_roadmap(user_query):
    # STEP A: Intent Extraction
    # We ask Gemini to turn the user's chat into structured data
    prompt_intent = f"""
    Extract the following entities from this user query in JSON format:
    Query: "{user_query}"
    Entities: Business_Type, Location, User_Category (Student/Entrepreneur/etc).
    """
    intent_response = model_gemini.generate_content(prompt_intent)
    print(f"--- Detected Intent ---\n{intent_response.text}")

    # STEP B: Semantic Search in your Local DB
    # We turn the query into a vector and find the best matches in your 120+ chunks
    query_vector = embedding_model.encode(user_query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3  # Get the top 3 most relevant policy chunks
    )
    
    context_data = "\n".join(results['documents'][0])

    # STEP C: Final Structured Roadmap
    # We give the law chunks to Gemini and ask it to make a simple roadmap
    prompt_roadmap = f"""
    You are PolicyNav, a legal assistant for marginalized communities in Tamil Nadu.
    Using ONLY the context below, provide a structured roadmap for: "{user_query}"
    
    Context from Government Records:
    {context_data}
    
    Format:
    1. Compliance (Licenses needed)
    2. Schemes (Subsidies eligible)
    3. Document Checklist
    """
    
    final_output = model_gemini.generate_content(prompt_roadmap)
    return final_output.text

if __name__ == "__main__":
    query = "I want to start a small business in Chennai, I am an unemployed youth."
    print("\n--- Generating Your PolicyNav Roadmap ---")
    print(get_policy_roadmap(query))