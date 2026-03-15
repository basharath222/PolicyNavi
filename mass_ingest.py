import asyncio
import uuid
import chromadb
import random
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize DB and Model
client = chromadb.PersistentClient(path="./policynav_db")
collection = client.get_or_create_collection(name="tamilnadu_schemes")
model = SentenceTransformer("all-MiniLM-L6-v2")

async def scrape_current_page(page, scheme_name):
    """Scrapes the tabs of the scheme page that is ALREADY open."""
    tabs = ["Details", "Benefits", "Eligibility", "Application Process", "Documents Required"]
    scheme_data = {}
    
    for tab in tabs:
        try:
            tab_btn = page.get_by_role("tab", name=tab)
            if await tab_btn.is_visible():
                await tab_btn.hover()
                await tab_btn.click()
                await asyncio.sleep(random.uniform(1.0, 2.0))
                content = await page.locator(".tab-content").inner_text()
                scheme_data[tab] = content
        except Exception as e:
            print(f"      Warning: Could not scrape tab {tab}: {e}")
            
    # Save to ChromaDB immediately after scraping all tabs for this scheme
    if scheme_data:
        for tab_name, text in scheme_data.items():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                embedding = model.encode(chunk).tolist()
                collection.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas={
                        "scheme_name": scheme_name, 
                        "tab": tab_name, 
                        "state": "Tamil Nadu"
                    }
                )
        return True
    return False

async def run_mass_scrape(target_count=234): 
    async with async_playwright() as p:
        # Headed mode is essential to bypass the current 'Sign Out' wall manually if needed
        browser = await p.chromium.launch(headless=False) 
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        stealth = Stealth()
        await stealth.apply_stealth_async(page)

        print("--- Navigating to Tamil Nadu Scheme Grid ---")
        await page.goto("https://www.myscheme.gov.in/search/state/Tamil%20Nadu", wait_until="networkidle")
        
        total_indexed = 0
        current_page_num = 1

        while total_indexed < target_count:
            # 1. Find all clickable scheme links on the current page
            # We use a locator that targets the titles
            await page.wait_for_selector("a[href*='/schemes/']", timeout=60000)
            scheme_locators = page.locator("a[href*='/schemes/']")
            count_on_page = await scheme_locators.count()
            
            print(f"\n--- Page {current_page_num}: Found {count_on_page} schemes ---")

            for i in range(count_on_page):
                try:
                    # Re-locate the card to avoid stale element errors
                    card = page.locator("a[href*='/schemes/']").nth(i)
                    scheme_name = await card.inner_text()
                    scheme_name = scheme_name.split('\n')[0] # Clean name
                    
                    print(f"  [{total_indexed + 1}] Opening: {scheme_name}")
                    
                    # Click and wait for the scheme page to load
                    await card.click()
                    await page.wait_for_load_state("networkidle")
                    await asyncio.sleep(random.uniform(2, 4))
                    
                    # Scrape and Save
                    success = await scrape_current_page(page, scheme_name)
                    if success:
                        print(f"    Successfully Indexed: {scheme_name}")
                        total_indexed += 1
                    
                    # Go back to the grid
                    await page.go_back()
                    await page.wait_for_selector("a[href*='/schemes/']", timeout=60000)
                    await asyncio.sleep(2)

                except Exception as e:
                    print(f"    Error scraping card {i}: {e}. Returning to grid...")
                    await page.goto("https://www.myscheme.gov.in/search/state/Tamil%20Nadu")
                    await asyncio.sleep(5)

            # 2. Pagination: Move to the next page number
            current_page_num += 1
            next_page_btn = page.locator(f"li:has-text('{current_page_num}')").first
            
            if await next_page_btn.is_visible():
                print(f"--- Moving to Page {current_page_num} ---")
                await next_page_btn.click()
                await asyncio.sleep(5)
            else:
                print("--- No more pages found. Scraping Complete. ---")
                break

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_mass_scrape())