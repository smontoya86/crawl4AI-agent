-- SQLite schema for URL tracking
CREATE TABLE IF NOT EXISTS crawl_urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed, removed
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    last_attempted TIMESTAMP,
    completed_at TIMESTAMP,
    last_seen TIMESTAMP,  -- Track when URL was last found in sitemap
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_status ON crawl_urls(status);
CREATE INDEX IF NOT EXISTS idx_url ON crawl_urls(url); 