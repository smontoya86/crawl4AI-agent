import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree
import gzip
from io import BytesIO

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Updated browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu", 
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-robots",  # Ignore robots.txt
        ],
        timeout=30000,  # Moved timeout here where it's supported
    )
    
    # Simplified crawler config
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    success_count += 1
                else:
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")

def get_weedmaps_urls():
    """
    Fetches all URLs from the Weedmaps sitemap index and its sub-sitemaps.
    Uses the sitemap (https://weedmaps.com/sitemap.xml.gz) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """            
    sitemap_index_url = "https://weedmaps.com/sitemap.xml.gz"
    all_urls = []
    
    try:
        # Fetch and parse the sitemap index
        response = requests.get(sitemap_index_url)
        response.raise_for_status()
        
        # Decompress the gzipped content
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
            content = gz_file.read()
        
        # Parse the XML
        root = ElementTree.fromstring(content)
        
        # Extract sitemap URLs from the index
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        sitemap_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        print(f"Found {len(sitemap_urls)} sitemaps to process")
        
        # Process each sitemap
        for sitemap_url in sitemap_urls:
            try:
                print(f"Processing sitemap: {sitemap_url}")
                response = requests.get(sitemap_url)
                response.raise_for_status()
                
                # Handle gzipped sitemaps
                if sitemap_url.endswith('.gz'):
                    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
                        content = gz_file.read()
                else:
                    content = response.content
                
                # Parse the sitemap XML
                sitemap_root = ElementTree.fromstring(content)
                urls = [loc.text for loc in sitemap_root.findall('.//ns:loc', namespace)]
                all_urls.extend(urls)
                print(f"Added {len(urls)} URLs from {sitemap_url}")
                
            except Exception as e:
                print(f"Error processing sitemap {sitemap_url}: {e}")
                continue
        
        print(f"Total URLs found across all sitemaps: {len(all_urls)}")
        return all_urls
        
    except Exception as e:
        print(f"Error fetching sitemap index: {e}")
        return []

async def main():
    urls = get_weedmaps_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        # Increased max_concurrent for faster crawling, adjust as needed
        await crawl_parallel(urls, max_concurrent=5)
    else:
        print("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())
