import asyncio
import uuid
import chromadb
from playwright.async_api import async_playwright
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize DB and Model
client = chromadb.PersistentClient(path="./policynav_db")
collection = client.get_or_create_collection(name="tamilnadu_schemes")
model = SentenceTransformer("all-MiniLM-L6-v2")

async def scrape_scheme_details(page, url):
    """Deep scrapes individual scheme tabs for contact and process info."""
    try:
        await page.goto(url, wait_until="networkidle")
        # Specific desktop heading to avoid strict mode errors
        name = await page.get_by_test_id("desktop-view-container").locator("h1").first.inner_text()
        
        scheme_data = {}
        tabs = ["Details", "Benefits", "Eligibility", "Application Process", "Documents Required"]
        
        for tab in tabs:
            tab_btn = page.get_by_role("tab", name=tab)
            if await tab_btn.is_visible():
                await tab_btn.click()
                await asyncio.sleep(1) # Wait for tab animation
                content = await page.locator(".tab-content").inner_text()
                scheme_data[tab] = content
        
        return name, scheme_data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

async def run_mass_scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="PolicyNavBot/2.0")
        page = await context.new_page()

        # Start at the Tamil Nadu state page
        base_url = "https://www.myscheme.gov.in/search/state/Tamil%20Nadu"
        await page.goto(base_url, wait_until="networkidle")

        all_scheme_links = []
        page_num = 1

        print("--- Phase A: Collecting all Scheme Links ---")
        while True:
            # Collect links on current page
            links = await page.eval_on_selector_all(
                "a[href^='/schemes/']", 
                "elements => elements.map(e => e.href)"
            )
            all_scheme_links.extend(list(set(links)))
            print(f"Page {page_num}: Found {len(links)} links. Total so far: {len(all_scheme_links)}")

            # Handle Pagination: Look for the 'Next' button
            # Note: myScheme uses a '>' or 'Next' icon in its pagination bar
            
            next_button = page.locator("button.pagination-next, li.next > a, [aria-label='Next page']").first
            if await next_button.is_visible() and await next_button.is_enabled():
                await next_button.click()
                # Wait for the new list of schemes to load
                await page.wait_for_selector("a[href^='/schemes/']") 
                page_num += 1
            else:
                break # No more pages

        print(f"--- Phase B: Deep Scraping {len(all_scheme_links)} Schemes ---")
        # To avoid being blocked, we process them sequentially or in small batches
        for link in all_scheme_links:
            name, data = await scrape_scheme_details(page, link)
            if name and data:
                # Store each tab as a unique chunk with full metadata
                for tab_name, text in data.items():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = text_splitter.split_text(text)
                    
                    for chunk in chunks:
                        embedding = model.encode(chunk).tolist()
                        collection.add(
                            ids=[str(uuid.uuid4())],
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas={
                                "scheme_name": name, 
                                "tab": tab_name, 
                                "source": link,
                                "state": "Tamil Nadu"
                            }
                        )
                print(f"  Successfully indexed: {name}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_mass_scrape())