import time
import uuid
import os
import chromadb
import fitz  # PyMuPDF
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize Local Persistent ChromaDB
# Data will be saved to a 'policynav_db' folder in your directory
client = chromadb.PersistentClient(path="./policynav_db")
collection = client.get_or_create_collection(name="tamilnadu_schemes")

# 2. Load Embedding Model (Runs locally for free)
print("Loading AI Embedding Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def save_to_chroma(text, name, tab_type, url):
    """Chunks text and stores it as vectors in ChromaDB."""
    # Split text into 700-character pieces with overlap to preserve context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    for chunk in chunks:
        # Convert text chunk into a mathematical vector
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk],
            metadatas={"scheme_name": name, "tab": tab_type, "source": url}
        )
    print(f"  -> Indexed {len(chunks)} chunks for {name} ({tab_type})")

def start_phase_1():
    # A. Process MSME PDF Booklet
    print("\n--- Processing Local MSME Booklet ---")
    pdf_filename = "Scheme-booklet-Eng.pdf"
    if os.path.exists(pdf_filename):
        try:
            doc = fitz.open(pdf_filename)
            pdf_text = "".join([page.get_text() for page in doc])
            save_to_chroma(pdf_text, "MSME Booklet", "Full Details", "Local PDF")
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"File {pdf_filename} not found. Skipping PDF step.")

    # B. Scrape Tamil Nadu Schemes from myScheme
    print("\n--- Scraping Tamil Nadu Schemes from myScheme ---")
    with sync_playwright() as p:
        # Launch browser (headless=True for speed)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="PolicyNavBot/1.0")
        
        # Navigate directly to Tamil Nadu search results
        tn_url = "https://www.myscheme.gov.in/search/state/Tamil%20Nadu"
        print(f"Visiting: {tn_url}")
        page.goto(tn_url, wait_until="networkidle")
        
        # Capture links for individual schemes (Limiting to 5 for the MVP test)
        links = page.eval_on_selector_all("a[href^='/schemes/']", "elements => elements.map(e => e.href)")
        unique_links = list(set(links))[:5] 
        print(f"Found {len(unique_links)} unique schemes to process.")

        for link in unique_links:
            try:
                page.goto(link, wait_until="networkidle")
                
                # FIX: Target the desktop-view heading to resolve multi-element errors
                name = page.get_by_test_id("desktop-view-container").locator("h1").first.inner_text()
                print(f"\nScraping Scheme: {name}")
                
                # Iterate through the specific tabs: Details, Benefits, Eligibility
                for tab in ["Details", "Benefits", "Eligibility"]:
                    tab_btn = page.get_by_role("tab", name=tab)
                    if tab_btn.is_visible():
                        tab_btn.click()
                        time.sleep(1.5) # Wait for content animation
                        content = page.locator(".tab-content").inner_text()
                        save_to_chroma(content, name, tab, link)
            except Exception as e:
                print(f"Skipping {link} due to error: {e}")
        
        browser.close()
    print("\nPhase 1 Complete! Knowledge base stored in './policynav_db'")

if __name__ == "__main__":
    start_phase_1()