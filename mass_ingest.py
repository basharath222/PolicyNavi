import asyncio
import uuid
import chromadb
import random
import os
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List of user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
]

# Free proxy list (these change frequently, you may need to update them)
PROXIES = [
    None,  # Try without proxy first
    # Add free proxies from https://free-proxy-list.net/ if needed
]

# Initialize DB and Model
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./policynav_db")

# Delete existing collection if it exists
try:
    client.delete_collection("tamilnadu_schemes")
    print("Deleted existing collection")
except:
    print("No existing collection to delete")

# Create fresh collection
collection = client.create_collection(name="tamilnadu_schemes")
print("Created new collection")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")

async def scrape_scheme_page(page, scheme_name, scheme_url):
    """Scrapes all tabs from a scheme page with improved selectors"""
    print(f"  📋 Scraping: {scheme_name}")
    
    tabs = ["Details", "Benefits", "Eligibility", "Application Process", "Documents Required"]
    scheme_data = {}
    
    for tab in tabs:
        try:
            # Try different selectors for tabs
            tab_selectors = [
                f"button:has-text('{tab}')",
                f"a:has-text('{tab}')",
                f"div[role='tab']:has-text('{tab}')",
                f"li:has-text('{tab}')"
            ]
            
            tab_found = False
            for selector in tab_selectors:
                tab_element = page.locator(selector).first
                if await tab_element.count() > 0 and await tab_element.is_visible():
                    await tab_element.click()
                    await asyncio.sleep(random.uniform(1, 2))
                    tab_found = True
                    break
            
            if not tab_found:
                print(f"    ⚠️ Tab '{tab}' not found")
                continue
            
            # Try different content selectors
            content_selectors = [
                ".tab-content",
                "[role='tabpanel']",
                ".scheme-content",
                ".details-content",
                "div[class*='content']",
                "div[class*='details']"
            ]
            
            content = None
            for selector in content_selectors:
                content_element = page.locator(selector).first
                if await content_element.count() > 0:
                    content = await content_element.inner_text()
                    if content and len(content.strip()) > 20:
                        break
            
            if content and len(content.strip()) > 20:
                scheme_data[tab] = content.strip()
                print(f"    ✅ Scraped {tab}: {len(content)} chars")
            else:
                print(f"    ⚠️ No content found for {tab}")
                
        except Exception as e:
            print(f"    ⚠️ Error scraping {tab}: {str(e)[:50]}")
    
    return scheme_data

async def process_scheme(page, scheme_name, scheme_url, total_indexed):
    """Process a single scheme"""
    print(f"\n  [{total_indexed + 1}] Processing: {scheme_name}")
    
    try:
        # Navigate to scheme page with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await page.goto(scheme_url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(random.uniform(3, 5))
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"    ⚠️ Retry {attempt + 1}/{max_retries} for {scheme_name}")
                await asyncio.sleep(random.uniform(5, 10))
        
        # Scrape the scheme
        scheme_data = await scrape_scheme_page(page, scheme_name, scheme_url)
        
        if scheme_data:
            # Save to ChromaDB
            chunks_saved = 0
            for tab_name, text in scheme_data.items():
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                
                for chunk in chunks:
                    if len(chunk.strip()) < 20:
                        continue
                        
                    # Generate embedding
                    embedding = model.encode(chunk).tolist()
                    
                    # Add to ChromaDB
                    collection.add(
                        ids=[str(uuid.uuid4())],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "scheme_name": scheme_name,
                            "scheme_url": scheme_url,
                            "tab": tab_name,
                            "state": "Tamil Nadu",
                            "chunk_length": len(chunk)
                        }]
                    )
                    chunks_saved += 1
            
            print(f"    ✅ Saved {chunks_saved} chunks for {scheme_name}")
            return True
        else:
            print(f"    ❌ No data scraped for {scheme_name}")
            return False
            
    except Exception as e:
        print(f"    ❌ Error processing {scheme_name}: {str(e)[:100]}")
        return False

async def get_scheme_links(page):
    """Extract all scheme links from current page"""
    try:
        # Wait for scheme links to load
        await page.wait_for_selector("a[href*='/schemes/']", timeout=15000)
        
        # Get all scheme links
        scheme_elements = await page.query_selector_all("a[href*='/schemes/']")
        
        schemes = []
        for elem in scheme_elements:
            try:
                href = await elem.get_attribute("href")
                title = await elem.inner_text()
                if href and title and '/schemes/' in href:
                    full_url = f"https://www.myscheme.gov.in{href}" if href.startswith('/') else href
                    schemes.append({
                        'name': title.strip().split('\n')[0],
                        'url': full_url
                    })
            except:
                continue
        
        # Remove duplicates
        seen = set()
        unique_schemes = []
        for s in schemes:
            if s['url'] not in seen:
                seen.add(s['url'])
                unique_schemes.append(s)
        
        return unique_schemes
    except Exception as e:
        print(f"Error getting scheme links: {e}")
        return []

async def run_mass_scrape():
    """Main scraping function"""
    print("\n" + "="*60)
    print("🚀 Starting PolicyNav Mass Scraper for Tamil Nadu Schemes")
    print("="*60)
    
    async with async_playwright() as p:
        # Launch browser with additional arguments to avoid detection
        print("\n🌐 Launching browser...")
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--start-maximized'
            ]
        )
        
        # Create context with random user agent
        user_agent = random.choice(USER_AGENTS)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=user_agent,
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
        )
        
        page = await context.new_page()
        
        # Navigate to Tamil Nadu schemes page with retry logic
        print("\n📱 Navigating to Tamil Nadu schemes page...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await page.goto(
                    "https://www.myscheme.gov.in/search/state/Tamil%20Nadu", 
                    wait_until="domcontentloaded",
                    timeout=60000
                )
                await asyncio.sleep(random.uniform(5, 10))
                
                # Check if we got blocked
                page_content = await page.content()
                if "Access Denied" in page_content or "blocked" in page_content.lower():
                    print(f"⚠️ Access denied on attempt {attempt + 1}, retrying with different settings...")
                    await asyncio.sleep(random.uniform(10, 20))
                    continue
                else:
                    print("✅ Successfully loaded page")
                    break
                    
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:50]}")
                if attempt == max_retries - 1:
                    print("❌ Failed to load page after multiple attempts")
                    await browser.close()
                    return
                await asyncio.sleep(random.uniform(10, 20))
        
        # Collect all scheme links
        all_schemes = []
        current_page = 1
        max_pages = 10
        
        print("\n🔍 Collecting scheme links from all pages...")
        
        while current_page <= max_pages:
            print(f"\n--- Page {current_page} ---")
            
            # Random delay between page loads
            await asyncio.sleep(random.uniform(3, 7))
            
            # Get schemes from current page
            page_schemes = await get_scheme_links(page)
            print(f"Found {len(page_schemes)} schemes on page {current_page}")
            
            all_schemes.extend(page_schemes)
            
            # Try to go to next page
            try:
                # Look for next page button
                next_selectors = [
                    f"a:has-text('{current_page + 1}')",
                    "a:has-text('Next')",
                    "a:has-text('›')",
                    "button:has-text('Next')",
                    "[aria-label='Next']"
                ]
                
                next_found = False
                for selector in next_selectors:
                    next_button = page.locator(selector).first
                    if await next_button.count() > 0 and await next_button.is_visible():
                        await next_button.click()
                        await asyncio.sleep(random.uniform(3, 5))
                        await page.wait_for_load_state("domcontentloaded")
                        next_found = True
                        break
                
                if next_found:
                    current_page += 1
                else:
                    print("No more pages found")
                    break
                    
            except Exception as e:
                print(f"Error navigating to next page: {e}")
                break
        
        # Remove duplicates
        seen_urls = set()
        unique_schemes = []
        for s in all_schemes:
            if s['url'] not in seen_urls:
                seen_urls.add(s['url'])
                unique_schemes.append(s)
        
        print(f"\n📊 Total unique schemes found: {len(unique_schemes)}")
        
        if len(unique_schemes) == 0:
            print("\n❌ No schemes found! The website might be blocking scraping.")
            print("\nAlternative approaches:")
            print("1. Try running during off-peak hours")
            print("2. Use a VPN to change your IP")
            print("3. Download the pre-scraped data from GitHub [citation:1]")
            await browser.close()
            return
        
        # Process each scheme
        successful = 0
        failed = 0
        
        print("\n" + "="*60)
        print("📥 Starting to scrape individual schemes")
        print("="*60)
        
        for idx, scheme in enumerate(unique_schemes):
            print(f"\n📌 Progress: {idx + 1}/{len(unique_schemes)}")
            
            success = await process_scheme(
                page, 
                scheme['name'], 
                scheme['url'], 
                successful
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Longer random delay between schemes to avoid rate limiting
            await asyncio.sleep(random.uniform(5, 10))
        
        await browser.close()
        
        # Final report
        print("\n" + "="*60)
        print("✅ SCRAPING COMPLETE!")
        print("="*60)
        print(f"📊 Final Statistics:")
        print(f"   • Total schemes found: {len(unique_schemes)}")
        print(f"   • Successfully scraped: {successful}")
        print(f"   • Failed: {failed}")
        
        # Verify database
        count = collection.count()
        print(f"\n💾 Database Status:")
        print(f"   • Total chunks in DB: {count}")

if __name__ == "__main__":
    asyncio.run(run_mass_scrape())