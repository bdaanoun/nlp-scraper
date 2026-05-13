import requests
from bs4 import BeautifulSoup
import sqlite3
import uuid
from datetime import datetime
import time

# CONFIGURATION

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/sport/rss.xml",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DB_PATH = "data/news.db"

# DATABASE SETUP

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id          TEXT PRIMARY KEY,
            url         TEXT UNIQUE,
            date        TEXT,
            headline    TEXT,
            body        TEXT
        )
    """)
    conn.commit()
    conn.close()

# PARSE RSS FEED

def parse_rss(feed_url):
    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "xml")

        articles = []
        for item in soup.find_all("item"):
            title   = item.find("title")
            link    = item.find("link")
            pubdate = item.find("pubDate")

            # Skip incomplete items
            if not title or not link:
                continue

            articles.append({
                "headline": title.get_text(strip=True),
                "url":      link.get_text(strip=True),
                "date":     pubdate.get_text(strip=True) if pubdate else datetime.now().isoformat(),
            })

        return articles

    except Exception as e:
        print(f"  [ERROR] Could not parse RSS {feed_url}: {e}")
        return []

# FETCH ARTICLE BODY

def fetch_body(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
            body = " ".join(p.get_text(strip=True) for p in paragraphs)
        else:
            body = None

        # Ignore articles with too little content
        if not body or len(body) < 100:
            return None

        return body

    except Exception as e:
        print(f"  [ERROR] Could not fetch body {url}: {e}")
        return None

# SAVE TO DATABASE

def save_article(article):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO articles (id, url, date, headline, body)
            VALUES (:id, :url, :date, :headline, :body)""", article)
        conn.commit()
        inserted = cursor.rowcount
    except Exception as e:
        print(f"[DB ERROR] {e}")
        inserted = 0
    finally:
        conn.close()
    return inserted

# COUNT ARTICLES

def count_articles():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# MAIN LOOP
def main():
    init_db()

    for feed_url in RSS_FEEDS:
        print(f"\nReading RSS feed: {feed_url}")
        items = parse_rss(feed_url)
        print(f"Found {len(items)} articles in feed")

        for item in items:
            print(f"\n{count_articles() + 1}. scraping {item['url']}")
            print("requesting ...")

            body = fetch_body(item["url"])

            if body:
                print("parsing ...")
                article = {
                    "id":       str(uuid.uuid4()),
                    "url":      item["url"],
                    "date":     item["date"],
                    "headline": item["headline"],
                    "body":     body,
                }
                saved = save_article(article)
                if saved:
                    print(f"saved in {DB_PATH}")
                else:
                    print("[SKIPPED] already in database")
            else:
                print("[SKIPPED] not enough content")


            time.sleep(1)
    print(f"\nDone! Total articles in DB: {count_articles()}")

if __name__ == "__main__":
    main()