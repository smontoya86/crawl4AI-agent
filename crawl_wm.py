import os
import sys
import json
import asyncio
import requests
import sqlite3
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv
import psutil
import gzip
from io import BytesIO
import aiosqlite
from pathlib import Path
import aiohttp
import random
from bs4 import BeautifulSoup
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

# Debug env file location
current_dir = Path(__file__).parent
env_file = current_dir / '.env'

print(f"Looking for .env file at: {env_file}")
print(f"File exists: {env_file.exists()}")

# Force reload environment
if env_file.exists():
    # Read the file directly first to debug
    with open(env_file, 'r') as f:
        env_contents = f.read()
        print("\nEnvironment file contents:")
        print(env_contents)
    
    # Force reload
    load_dotenv(env_file, override=True)
    print("\nLoaded .env file successfully")
else:
    print("ERROR: Could not find .env file")

# Debug environment variables
api_key = os.getenv("OPENAI_API_KEY")
print(f"\nEnvironment variable details:")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"API Key prefix: {api_key[:7]}..." if api_key else "No key")

# Validate OpenAI API key format
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
elif not api_key.startswith(('sk-', 'sk-proj-')):
    raise ValueError(f"Invalid OPENAI_API_KEY format. Should start with 'sk-' or 'sk-proj-'. Found: {api_key[:7]}...")
elif len(api_key) < 20:  # OpenAI keys are typically much longer
    raise ValueError(f"OPENAI_API_KEY seems too short (length: {len(api_key)})")

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(
    api_key=api_key
)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Move system prompt to module level
SYSTEM_PROMPT = """You are an AI that extracts titles and summaries from Weedmaps content.
Return a JSON object with 'title' and 'summary' keys.
For the title: If this seems like the start of a page, extract its title. If it's a middle chunk, derive a descriptive title.
For the summary: Create a concise summary of the main points in this chunk, focusing on dispensary, product, or cannabis-related information.
Keep both title and summary concise but informative."""

# Move browser_config to module level
BROWSER_CONFIG = BrowserConfig(
    headless=True,
    verbose=True,
    extra_args=[
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--window-size=1920,1080",
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
        "--enable-javascript",
        "--no-zygote",
        "--no-first-run",
        "--disable-setuid-sandbox"
    ]
)

# Move crawl_config to module level
CRAWL_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from Weedmaps content.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a page, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk, focusing on dispensary, product, or cannabis-related information.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "WM",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunks_batch(chunks: List[str], url: str) -> List[ProcessedChunk]:
    """Process multiple chunks efficiently in batches."""
    try:
        # Clean and validate input texts
        texts_to_embed = []
        for chunk in chunks:
            # Debug the incoming chunk
            print(f"Debug - Chunk type: {type(chunk)}, length: {len(str(chunk))}")
            print(f"Debug - First 100 chars: {str(chunk)[:100]}")
            
            cleaned_text = str(chunk).strip()
            if cleaned_text:
                # Truncate if needed
                cleaned_text = cleaned_text[:8000]  # OpenAI limit
                texts_to_embed.append(cleaned_text)
            else:
                print(f"Empty chunk found for {url}")
                texts_to_embed.append("No content available")
        
        if not texts_to_embed:
            print(f"No valid texts to embed for {url}")
            return []
            
        # Make the API call with validated input
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts_to_embed
        )
        embeddings = [e.embedding for e in embedding_response.data]
        
        if len(embeddings) != len(chunks):
            print(f"Warning: Mismatch in embeddings count for {url} - got {len(embeddings)}, expected {len(chunks)}")
            # Pad with zero vectors if necessary
            embeddings.extend([[0] * 1536] * (len(chunks) - len(embeddings)))
            
    except Exception as e:
        print(f"Error getting embeddings batch for {url}: {e}")
        print(f"Input texts sample: {texts_to_embed[0][:100] if texts_to_embed else 'No texts'}")
        embeddings = [[0] * 1536 for _ in chunks]  # Zero vectors for errors
    
    # Get titles and summaries in batch (max 20 per batch for chat completions)
    processed_chunks = []
    for i in range(0, len(chunks), 20):
        batch = chunks[i:i+20]
        batch_embeddings = embeddings[i:i+20]
        
        try:
            # Create messages for batch
            messages = []
            for chunk in batch:
                messages.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ])
            
            # Get summaries in parallel
            responses = await asyncio.gather(*[
                openai_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    messages=msg,
                    response_format={ "type": "json_object" }
                ) for msg in messages
            ])
            
            summaries = [json.loads(r.choices[0].message.content) for r in responses]
            
            # Create ProcessedChunks
            for chunk, embedding, summary in zip(batch, batch_embeddings, summaries):
                processed_chunks.append(ProcessedChunk(
                    url=url,
                    chunk_number=len(processed_chunks),
                    title=summary['title'],
                    summary=summary['summary'],
                    content=chunk,
                    metadata={
                        "source": "WM",
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "url_path": urlparse(url).path
                    },
                    embedding=embedding
                ))
                
        except Exception as e:
            print(f"Error processing batch for {url}: {e}")
            # Add error placeholders
            for chunk in batch:
                processed_chunks.append(ProcessedChunk(
                    url=url,
                    chunk_number=len(processed_chunks),
                    title="Error processing title",
                    summary="Error processing summary",
                    content=chunk,
                    metadata={
                        "source": "WM",
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "url_path": urlparse(url).path,
                        "error": str(e)
                    },
                    embedding=[0] * 1536
                ))
    
    return processed_chunks

async def extract_urls_from_markdown(content: str, base_url: str, patterns: List[str]) -> Set[str]:
    """Extract URLs from markdown content."""
    urls = set()
    
    # Skip image and asset URLs
    skip_patterns = [
        '.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico',
        '.css', '.js', '.woff', '.ttf',
        'files.readme.io',
        'assets.',
        'cdn.',
        'static.'
    ]
    
    # Extract markdown links [text](url)
    markdown_links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
    for text, href in markdown_links:
        # Skip if it's an image or asset
        if any(pattern in href.lower() for pattern in skip_patterns):
            continue
            
        # Clean the URL
        href = href.split('#')[0]  # Remove fragment
        href = href.split('?')[0]  # Remove query params
        
        if href.startswith('/'):
            href = f"{base_url.rstrip('/')}{href}"
        elif not href.startswith('http'):
            href = f"{base_url.rstrip('/')}/{href}"
        
        # Check if URL matches patterns
        if any(pattern in href for pattern in patterns):
            urls.add(href)
    
    # Extract raw URLs
    raw_urls = re.findall(r'https?://[^\s<>"\']+', content)
    for href in raw_urls:
        # Skip if it's an image or asset
        if any(pattern in href.lower() for pattern in skip_patterns):
            continue
            
        href = href.split('#')[0]
        href = href.split('?')[0]
        if any(pattern in href for pattern in patterns):
            urls.add(href)
    
    return urls

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    try:
        # Skip if content is too short or empty
        if not markdown or len(markdown.strip()) < 100:
            print(f"Warning: Empty or very short content for {url}")
            return False
            
        # Check for bot detection/error pages
        bot_markers = [
            "detected unusual activity",
            "please verify you are a human",
            "access to this page has been denied",
            "complete the security check",
            "image not found",
            "404 not found",
            "page not found"
        ]
            
        if any(marker in markdown.lower() for marker in bot_markers):
            print(f"Invalid content detected for {url}")
            return False

        # Clean the markdown content
        cleaned_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown)  # Remove markdown links
        cleaned_content = re.sub(r'!\[([^\]]+)\]\([^)]+\)', '', cleaned_content)  # Remove images
        cleaned_content = re.sub(r'#{1,6}\s+', '', cleaned_content)  # Remove headers
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)  # Normalize whitespace
        
        # Split into chunks
        chunks = chunk_text(cleaned_content)
        
        # Process in smaller batches of 5 chunks
        for i in range(0, len(chunks), 5):
            batch = chunks[i:i+5]
            
            try:
                # Get embeddings for batch
                embedding_response = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    timeout=30  # 30 second timeout
                )
                embeddings = [e.embedding for e in embedding_response.data]
                
                # Get titles and summaries (one at a time to avoid rate limits)
                processed_chunks = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    try:
                        response = await asyncio.wait_for(
                            openai_client.chat.completions.create(
                                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                                ],
                                response_format={ "type": "json_object" },
                                timeout=30
                            ),
                            timeout=35
                        )
                        
                        result = json.loads(response.choices[0].message.content)
                        
                        processed_chunks.append({
                            "url": url,
                            "chunk_number": i + j,
                            "title": result['title'],
                            "summary": result['summary'],
                            "content": chunk,
                            "metadata": {
                                "source": "WM",
                                "chunk_size": len(chunk),
                                "crawled_at": datetime.now(timezone.utc).isoformat(),
                                "url_path": urlparse(url).path
                            },
                            "embedding": embedding
                        })
                        
                        # Small delay between API calls
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error processing chunk {i+j} for {url}: {e}")
                        continue
                
                # Store batch in Supabase
                if processed_chunks:
                    try:
                        result = supabase.table("site_pages").insert(processed_chunks).execute()
                        print(f"Stored {len(processed_chunks)} chunks for {url}")
                    except Exception as e:
                        print(f"Error storing chunks in Supabase: {e}")
                        return False
                        
            except Exception as e:
                print(f"Error processing batch for {url}: {e}")
                continue
                
            # Small delay between batches
            await asyncio.sleep(1)
            
        return True
        
    except Exception as e:
        print(f"Error in process_and_store_document for {url}: {e}")
        return False

class CrawlManager:
    def __init__(self, db_path: str = "weedmaps_crawl.db"):
        self.db_path = db_path
        self.batch_size = 1000
        self.max_retries = 3
        
    async def init_db(self):
        """Initialize the SQLite database with our schema."""
        async with aiosqlite.connect(self.db_path) as db:
            with open('sqlite_schema.sql', 'r') as f:
                schema = f.read()
            await db.executescript(schema)
            await db.commit()
    
    async def save_urls(self, urls: List[str]):
        """Save new URLs to the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """INSERT OR IGNORE INTO crawl_urls (url) VALUES (?)""",
                [(url,) for url in urls]
            )
            await db.commit()
    
    async def get_next_batch(self) -> List[str]:
        """Get the next batch of pending URLs."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT url FROM crawl_urls 
                WHERE status = 'pending' 
                AND (retry_count < ? OR retry_count IS NULL)
                LIMIT ?
                """,
                (self.max_retries, self.batch_size)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def update_status(self, url: str, status: str, error: Optional[str] = None):
        """Update the status of a URL."""
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc)
            if status == 'completed':
                await db.execute(
                    """
                    UPDATE crawl_urls 
                    SET status = ?, completed_at = ?, last_attempted = ?
                    WHERE url = ?
                    """,
                    (status, now, now, url)
                )
            else:
                await db.execute(
                    """
                    UPDATE crawl_urls 
                    SET status = ?, error = ?, 
                        retry_count = COALESCE(retry_count, 0) + 1,
                        last_attempted = ?
                    WHERE url = ?
                    """,
                    (status, error, now, url)
                )
            await db.commit()
    
    async def get_stats(self) -> Dict[str, int]:
        """Get crawling statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT status, COUNT(*) 
                FROM crawl_urls 
                GROUP BY status
                """
            )
            rows = await cursor.fetchall()
            return {status: count for status, count in rows}

    async def update_url_list(self, new_urls: List[str]):
        """
        Update the URL list by:
        1. Adding new URLs
        2. Marking removed URLs
        3. Identifying URLs for recrawl
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get existing URLs
            cursor = await db.execute("SELECT url, status FROM crawl_urls")
            existing = {url: status for url, status in await cursor.fetchall()}
            
            # Find new URLs
            new_urls_set = set(new_urls)
            existing_urls_set = set(existing.keys())
            
            # URLs to add (completely new)
            urls_to_add = new_urls_set - existing_urls_set
            if urls_to_add:
                print(f"Found {len(urls_to_add)} new URLs to add")
                # Process in chunks of 1000
                for i in range(0, len(urls_to_add), 1000):
                    chunk = list(urls_to_add)[i:i + 1000]
                    await db.executemany(
                        "INSERT INTO crawl_urls (url, status) VALUES (?, 'pending')",
                        [(url,) for url in chunk]
                    )
            
            # URLs that no longer exist
            removed_urls = existing_urls_set - new_urls_set
            if removed_urls:
                print(f"Found {len(removed_urls)} URLs that have been removed")
                # Process in chunks of 1000
                for i in range(0, len(removed_urls), 1000):
                    chunk = list(removed_urls)[i:i + 1000]
                    await db.executemany(
                        "UPDATE crawl_urls SET status = 'removed' WHERE url = ?",
                        [(url,) for url in chunk]
                    )
            
            # Reset failed URLs for retry - using a more efficient query
            await db.execute("""
                UPDATE crawl_urls 
                SET status = 'pending', retry_count = 0 
                WHERE status = 'failed'
            """)
            
            await db.commit()
            
            # Get statistics
            stats = await self.get_stats()
            print("\nURL Status Summary:")
            for status, count in stats.items():
                print(f"  {status}: {count}")
            
            return len(urls_to_add)

    async def reset_failed_urls(self):
        """Reset URLs that had empty or invalid content."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE crawl_urls 
                SET status = 'pending', 
                    retry_count = 0,
                    error = NULL 
                WHERE status = 'completed' 
                AND url NOT IN (
                    SELECT url FROM site_pages
                )
            """)
            await db.commit()

async def crawl_documentation_site(browser: AsyncWebCrawler, base_url: str, patterns: List[str], 
                                 max_depth: int = 3, start_url: str = None, date_cutoff: str = None) -> Set[str]:
    """Recursively crawl ganjapreneur.com to find all relevant URLs."""
    visited = set()
    to_visit = {start_url or base_url}
    found_urls = set()
    depth = 0
    
    skip_patterns = [
        '.jpg', '.jpeg', '.png', '.gif', '.css', '.js',
        'mailto:', 'javascript:', '#',
        '/wp-content/', '/wp-includes/',
        '/feed/', '/author/', '/tag/',
        '/page/', '/category/',
        'twitter.com', 'facebook.com', 'linkedin.com',
        '/comment', '/submit', '/contact',
        '/search', '/advertise', '/about'
    ]
    
    print(f"\nStarting crawl from {start_url or base_url}")
    
    while to_visit and depth < max_depth:
        current_urls = to_visit
        to_visit = set()
        depth += 1
        
        print(f"\nDepth {depth}: Processing {len(current_urls)} URLs")
        
        for url in current_urls:
            if url in visited:
                continue
            
            visited.add(url)
            print(f"\nCrawling: {url}")
            
            try:
                result = await browser.arun(url=url, config=CRAWL_CONFIG)
                if result.success:
                    content = result.markdown_v2.raw_markdown
                    print(f"\nRaw content sample:\n{content[:1000]}")  # Debug raw content
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    print("\nPage Structure:")
                    print(f"Title: {soup.title.string if soup.title else 'No title found'}")
                    
                    # Debug HTML structure
                    print("\nFound elements:")
                    print(f"<main> tags: {len(soup.find_all('main'))}")
                    print(f"<article> tags: {len(soup.find_all('article'))}")
                    print(f"<div class='post'> elements: {len(soup.find_all('div', class_='post'))}")
                    print(f"All links: {len(soup.find_all('a'))}")
                    
                    # Try multiple selectors for articles
                    article_containers = (
                        soup.find_all('article') or 
                        soup.find_all('div', class_='post') or
                        soup.find_all('div', class_='entry') or
                        soup.select('.post-item') or
                        soup.select('.news-item') or
                        soup.select('.article-content')
                    )
                    
                    if article_containers:
                        print(f"\nFound {len(article_containers)} article containers")
                        
                        for article in article_containers:
                            # Debug article content
                            print("\nArticle content:")
                            print(article.get_text()[:200])
                            
                            # Find links in this article
                            article_links = article.find_all('a', href=True)
                            print(f"Found {len(article_links)} links in article")
                            
                            for link in article_links:
                                href = link.get('href', '').strip()
                                if href:
                                    print(f"\nProcessing link: {href}")
                                    print(f"Link text: {link.text.strip()}")
                                    
                                    # Clean and normalize URL
                                    if href.startswith('/'):
                                        href = f"{base_url}{href}"
                                    elif not href.startswith('http'):
                                        href = f"{base_url}/{href}"
                                    elif not href.startswith(base_url):
                                        continue
                                    
                                    # Remove query parameters and fragments
                                    href = href.split('#')[0].split('?')[0]
                                    
                                    if any(p in href.lower() for p in skip_patterns):
                                        print("Skipping - matched skip pattern")
                                        continue
                                    
                                    if date_cutoff:
                                        year_match = re.search(r'/20(\d{2})/', href)
                                        if year_match:
                                            year = f"20{year_match.group(1)}"
                                            if year < date_cutoff:
                                                print("Skipping - before date cutoff")
                                                continue
                                    
                                    if any(pattern in href for pattern in patterns):
                                        print(f"Adding URL: {href}")
                                        found_urls.add(href)
                                        if href not in visited:
                                            to_visit.add(href)
                    else:
                        print("\nNo article containers found, trying direct link extraction")
                        # Try finding links directly
                        for link in soup.find_all('a', href=True):
                            href = link.get('href', '').strip()
                            if href and '/20' in href:  # Look for year in URL
                                print(f"\nFound potential article link: {href}")
                                if href.startswith('/'):
                                    href = f"{base_url}{href}"
                                elif not href.startswith('http'):
                                    href = f"{base_url}/{href}"
                                
                                if any(pattern in href for pattern in patterns):
                                    print(f"Adding URL: {href}")
                                    found_urls.add(href)
                                    if href not in visited:
                                        to_visit.add(href)
                    
                    # Look for pagination
                    pagination_selectors = [
                        '.pagination',
                        '.nav-links',
                        '.page-numbers',
                        'a.next',
                        'a.nextpostslink',
                        'a[rel="next"]'
                    ]
                    
                    for selector in pagination_selectors:
                        next_links = soup.select(selector)
                        if next_links:
                            print(f"\nFound pagination using selector: {selector}")
                            for link in next_links:
                                href = link.get('href', '').strip()
                                if href and not any(p in href.lower() for p in skip_patterns):
                                    if href.startswith('/'):
                                        href = f"{base_url}{href}"
                                    elif not href.startswith('http'):
                                        href = f"{base_url}/{href}"
                                    
                                    if href not in visited:
                                        to_visit.add(href)
                                        print(f"Added pagination URL: {href}")
                    
                    print(f"\nFound {len(found_urls)} total URLs so far")
                    print(f"To visit: {len(to_visit)} URLs")
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return found_urls

async def get_docs_urls() -> List[str]:
    """Get URLs from ganjapreneur.com sitemap."""
    all_urls = set()
    
    config = {
        'base_url': 'https://www.ganjapreneur.com',
        'sitemap_url': 'https://www.ganjapreneur.com/post-sitemap1.xml',
        'date_cutoff': '2022'  # Only get content from 2022 onwards
    }
    
    try:
        print(f"\nFetching sitemap from {config['sitemap_url']}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(config['sitemap_url']) as response:
                if response.status == 200:
                    sitemap_content = await response.text()
                    
                    # Parse XML sitemap
                    root = ElementTree.fromstring(sitemap_content)
                    
                    # Extract URLs and their last modified dates
                    for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                        loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                        lastmod = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                        
                        if loc is not None:
                            url_str = loc.text
                            
                            # Check if URL is recent enough based on lastmod
                            if lastmod is not None:
                                date_str = lastmod.text[:4]  # Get year from lastmod
                                if date_str >= config['date_cutoff']:
                                    all_urls.add(url_str)
                            else:
                                # If no lastmod, include URL
                                all_urls.add(url_str)
                
                print(f"Found {len(all_urls)} URLs in sitemap")
                return list(all_urls)
                
    except Exception as e:
        print(f"Error processing sitemap: {e}")
        import traceback
        traceback.print_exc()
        return []

    return list(all_urls)

async def main():
    # Initialize the crawl manager
    manager = CrawlManager()
    await manager.init_db()
    
    # Get URLs from Weedmaps
    urls = await get_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    # Update URL list and get count of new URLs
    print(f"Checking {len(urls)} URLs against database...")
    await manager.update_url_list(urls)
    
    # Get current stats
    stats = await manager.get_stats()
    pending_count = stats.get('pending', 0)
    
    if pending_count == 0:
        print("No URLs to process")
        return
    
    print(f"\nStarting to process {pending_count} pending URLs...")
    
    # Process in batches
    while True:
        batch = await manager.get_next_batch()
        if not batch:
            break
            
        print(f"\nProcessing batch of {len(batch)} URLs...")
        await crawl_parallel(batch, manager)
        
        # Show progress
        stats = await manager.get_stats()
        print("\nCurrent progress:")
        for status, count in stats.items():
            print(f"  {status}: {count}")

async def crawl_parallel(urls: List[str], manager: CrawlManager, max_concurrent: int = 15):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    # Use module-level config
    crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
    await crawler.start()
    
    # Track timing
    start_time = datetime.now()
    processed_count = 0
    
    print(f"\nStarting crawl of {len(urls)} URLs")
    
    # Track peak memory usage
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    try:
        success_count = 0
        fail_count = 0
        
        # Process URLs in batches
        for i in range(0, len(urls), max_concurrent):
            batch_start_time = datetime.now()
            current_batch = i // max_concurrent + 1
            batch = urls[i : i + max_concurrent]
            
            # Calculate time estimates
            processed_count = success_count + fail_count
            if processed_count > 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                urls_per_second = processed_count / elapsed_time
                remaining_urls = len(urls) - processed_count
                estimated_seconds_left = remaining_urls / urls_per_second
                estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds_left)
                
                print(f"\n=== Progress Update ===")
                print(f"Batch: {current_batch}/{len(urls) // max_concurrent + 1}")
                print(f"Total Progress: {processed_count}/{len(urls)} URLs in current set")
                print(f"Remaining URLs to process: {remaining_urls}")
                print(f"Success: {success_count}, Failed: {fail_count}")
                print(f"Processing rate: {urls_per_second:.2f} URLs/second")
                print(f"Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Estimated time remaining: {timedelta(seconds=int(estimated_seconds_left))}")
            
            tasks = []
            for j, url in enumerate(batch):
                session_id = f"parallel_session_{i + j}"
                await manager.update_status(url, 'processing')
                # Add timeout to crawler task
                task = asyncio.create_task(
                    asyncio.wait_for(
                        crawler.arun(url=url, config=CRAWL_CONFIG, session_id=session_id),
                        timeout=300  # 5 minute timeout per URL
                    )
                )
                tasks.append(task)

            # Log memory before batch
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            try:
                # Process batch with overall timeout
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=600  # 10 minute timeout for entire batch
                )

                # Process results and store documents
                for url, result in zip(batch, results):
                    if isinstance(result, (Exception, asyncio.TimeoutError)):
                        print(f"Error/Timeout crawling {url}: {result}")
                        await manager.update_status(url, 'failed', str(result))
                        fail_count += 1
                    elif result.success:
                        try:
                            # Debug and validate content
                            content = result.markdown_v2.raw_markdown
                            print(f"\nValidating content for {url}:")
                            print(f"Content length: {len(content)}")
                            
                            if await process_and_store_document(url, content):
                                await manager.update_status(url, 'completed')
                                success_count += 1
                            else:
                                await manager.update_status(url, 'failed', 'Invalid or blocked content')
                                fail_count += 1
                                
                            # Add random delay between requests (1-3 seconds)
                            await asyncio.sleep(random.uniform(1, 3))
                            
                        except Exception as e:
                            print(f"Error processing document {url}: {e}")
                            await manager.update_status(url, 'failed', str(e))
                            fail_count += 1
                    else:
                        await manager.update_status(url, 'failed', 'Unknown error')
                        fail_count += 1

                # Update progress after batch
                stats = await manager.get_stats()
                remaining_urls = stats.get('pending', 0)
                print(f"\nBatch {current_batch}/{len(urls) // max_concurrent + 1} complete.")
                print(f"Remaining URLs to process: {remaining_urls}")
                print(f"Success: {success_count}, Failed: {fail_count}")

            except asyncio.TimeoutError:
                print(f"\nTimeout processing batch {current_batch}/{len(urls) // max_concurrent + 1}")
                for url in batch:
                    await manager.update_status(url, 'failed', 'Batch timeout')
                fail_count += len(batch)

            # Reduced delay between batches
            await asyncio.sleep(0.25)

        print(f"\nFinal Summary:")
        print(f"  - Successfully crawled and processed: {success_count}")
        print(f"  - Failed: {fail_count}")

    except Exception as e:
        print(f"\nUnexpected error in crawl_parallel: {e}")
        raise
    finally:
        print("\nClosing crawler...")
        await crawler.close()
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")

if __name__ == "__main__":
    asyncio.run(main())
