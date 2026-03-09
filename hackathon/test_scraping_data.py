"""
GenZ Bible Scraper
==================
Scrapes https://genz.bible/ and saves each chapter as a JSON document
ready for ingestion into ChromaDB for RAG.

Requirements:
    pip install playwright beautifulsoup4
    python -m playwright install chromium

Output:
    genz_bible_chunks.json   — flat list of verse-level chunks for ChromaDB
    genz_bible_full.json     — full structured data (book → chapter → verses)

Usage:
    python scrape_genz_bible.py                  # scrape everything
    python scrape_genz_bible.py --books Matthew John  # scrape specific books
    python scrape_genz_bible.py --testaments NT  # scrape only New Testament
"""

import json
import time
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# Bible book metadata: name, chapter count, testament
# ─────────────────────────────────────────────
BOOKS = [
    # Old Testament
    ("Genesis",       50, "OT"), ("Exodus",        40, "OT"), ("Leviticus",      27, "OT"),
    ("Numbers",       36, "OT"), ("Deuteronomy",   34, "OT"), ("Joshua",         24, "OT"),
    ("Judges",        21, "OT"), ("Ruth",           4, "OT"), ("1-Samuel",       31, "OT"),
    ("2-Samuel",      24, "OT"), ("1-Kings",        22, "OT"), ("2-Kings",        25, "OT"),
    ("1-Chronicles",  29, "OT"), ("2-Chronicles",  36, "OT"), ("Ezra",           10, "OT"),
    ("Nehemiah",      13, "OT"), ("Esther",         10, "OT"), ("Job",            42, "OT"),
    ("Psalms",       150, "OT"), ("Proverbs",       31, "OT"), ("Ecclesiastes",  12, "OT"),
    ("Song-of-Solomon", 8, "OT"), ("Isaiah",        66, "OT"), ("Jeremiah",      52, "OT"),
    ("Lamentations",   5, "OT"), ("Ezekiel",        48, "OT"), ("Daniel",        12, "OT"),
    ("Hosea",         14, "OT"), ("Joel",            3, "OT"), ("Amos",           9, "OT"),
    ("Obadiah",        1, "OT"), ("Jonah",           4, "OT"), ("Micah",          7, "OT"),
    ("Nahum",          3, "OT"), ("Habakkuk",        3, "OT"), ("Zephaniah",      3, "OT"),
    ("Haggai",         2, "OT"), ("Zechariah",      14, "OT"), ("Malachi",        4, "OT"),
    # New Testament
    ("Matthew",       28, "NT"), ("Mark",           16, "NT"), ("Luke",           24, "NT"),
    ("John",          21, "NT"), ("Acts",           28, "NT"), ("Romans",         16, "NT"),
    ("1-Corinthians", 16, "NT"), ("2-Corinthians",  13, "NT"), ("Galatians",       6, "NT"),
    ("Ephesians",      6, "NT"), ("Philippians",     4, "NT"), ("Colossians",      4, "NT"),
    ("1-Thessalonians",5, "NT"), ("2-Thessalonians", 3, "NT"), ("1-Timothy",       6, "NT"),
    ("2-Timothy",      4, "NT"), ("Titus",           3, "NT"), ("Philemon",        1, "NT"),
    ("Hebrews",       13, "NT"), ("James",           5, "NT"), ("1-Peter",         5, "NT"),
    ("2-Peter",        3, "NT"), ("1-John",          5, "NT"), ("2-John",          1, "NT"),
    ("3-John",         1, "NT"), ("Jude",            1, "NT"), ("Revelation",     22, "NT"),
]

BASE_URL = "https://genz.bible"

# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def parse_chapter(html: str, book: str, chapter: int) -> list[dict]:
    """
    Extract verse-level chunks from rendered page HTML.
    Returns a list of dicts, each representing one verse.
    """
    soup = BeautifulSoup(html, "html.parser")
    chunks = []

    # genz.bible renders verses as <p> or <span> tags with data-verse attributes
    # Try multiple selectors to be resilient to layout changes
    verse_els = (
        soup.select("[data-verse]") or          # preferred: explicit verse markers
        soup.select(".verse") or                # fallback class
        soup.select("article p") or             # fallback article paragraphs
        soup.select("main p")                   # last resort
    )

    if not verse_els:
        # If no verse elements found, grab all paragraph text as one chunk
        text = soup.get_text(separator=" ", strip=True)
        if text:
            chunks.append(build_chunk(book, chapter, 0, text))
        return chunks

    for el in verse_els:
        verse_num = el.get("data-verse", 0)
        try:
            verse_num = int(verse_num)
        except (ValueError, TypeError):
            verse_num = 0

        text = el.get_text(separator=" ", strip=True)
        if text:
            chunks.append(build_chunk(book, chapter, verse_num, text))

    return chunks


def build_chunk(book: str, chapter: int, verse: int, text: str) -> dict:
    """Build a single RAG-ready chunk with metadata."""
    doc_id = f"{book}_{chapter}_{verse}" if verse else f"{book}_{chapter}"
    reference = f"{book} {chapter}:{verse}" if verse else f"{book} {chapter}"
    return {
        "id": doc_id,
        "reference": reference,
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "testament": None,   # filled in by scrape loop
        "text": text,
        "metadata": {
            "source": "genz.bible",
            "translation": "GenZ Translation",
        }
    }


# ─────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────

def scrape_all(books_filter=None, testaments_filter=None, delay=1.5):
    """
    Main scrape loop. Launches a single Playwright browser and iterates
    over all book/chapter combinations.

    Args:
        books_filter:     list of book names to scrape (None = all)
        testaments_filter: "OT", "NT", or None (= both)
        delay:            seconds to wait between requests (be polite!)
    """
    all_chunks = []
    full_data = {}
    errors = []

    # Filter books
    target_books = [
        (name, chapters, testament) for name, chapters, testament in BOOKS
        if (books_filter is None or name in books_filter)
        and (testaments_filter is None or testament == testaments_filter)
    ]

    total_chapters = sum(c for _, c, _ in target_books)
    print(f"📖 Scraping {len(target_books)} books / {total_chapters} chapters from genz.bible")
    print(f"⏱  Estimated time: ~{round(total_chapters * delay / 60, 1)} minutes\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (compatible; research-bot/1.0)"
        )
        page = context.new_page()

        scraped = 0
        for book_name, num_chapters, testament in target_books:
            full_data[book_name] = {"testament": testament, "chapters": {}}
            print(f"\n📕 {book_name} ({num_chapters} chapters)")

            for chapter_num in range(1, num_chapters + 1):
                url = f"{BASE_URL}/{book_name}/{chapter_num}"
                try:
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    html = page.content()
                    chunks = parse_chapter(html, book_name, chapter_num)

                    # Attach testament to each chunk
                    for chunk in chunks:
                        chunk["testament"] = testament

                    all_chunks.extend(chunks)
                    full_data[book_name]["chapters"][chapter_num] = chunks

                    scraped += 1
                    print(f"  ✅ Chapter {chapter_num:3d} — {len(chunks)} verses", end="\r")
                    time.sleep(delay)

                except Exception as e:
                    err_msg = f"{book_name} ch.{chapter_num}: {e}"
                    errors.append(err_msg)
                    print(f"  ❌ Chapter {chapter_num}: {e}")

            print(f"  ✅ {book_name} done — {sum(len(v) for v in full_data[book_name]['chapters'].values())} verses total")

        browser.close()

    print(f"\n\n✨ Done! Scraped {scraped} chapters, {len(all_chunks)} total chunks.")
    if errors:
        print(f"⚠️  {len(errors)} errors:")
        for e in errors:
            print(f"   - {e}")

    return all_chunks, full_data, errors


# ─────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────

def save_outputs(all_chunks: list, full_data: dict, output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Flat chunks file — ready for ChromaDB ingestion
    chunks_path = out / "genz_bible_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(all_chunks)} chunks → {chunks_path}")

    # 2. Full structured data
    full_path = out / "genz_bible_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved full structured data → {full_path}")

    # 3. Errors log
    return chunks_path, full_path


# ─────────────────────────────────────────────
# ChromaDB loader (bonus utility)
# ─────────────────────────────────────────────

def load_into_chromadb(chunks_path: str, collection_name: str = "genz_bible"):
    """
    Load scraped chunks into ChromaDB.
    Run this after scraping is complete.

    pip install chromadb
    """
    try:
        import chromadb
    except ImportError:
        print("ChromaDB not installed. Run: pip install chromadb")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Batch upsert in chunks of 500
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "reference": c["reference"],
                "book": c["book"],
                "chapter": c["chapter"],
                "verse": c["verse"],
                "testament": c["testament"] or "",
            } for c in batch],
        )
        print(f"  Loaded batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\n✅ Loaded {len(chunks)} chunks into ChromaDB collection '{collection_name}'")
    print(f"   Total in collection: {collection.count()}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape genz.bible for RAG")
    parser.add_argument("--books", nargs="+", help="Specific book names to scrape (e.g. Matthew John)")
    parser.add_argument("--testaments", choices=["OT", "NT"], help="Scrape only OT or NT")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests (default: 1.5)")
    parser.add_argument("--output-dir", default="./genz_bible_data", help="Output directory")
    parser.add_argument("--load-chromadb", action="store_true", help="Load into ChromaDB after scraping")
    args = parser.parse_args()

    all_chunks, full_data, errors = scrape_all(
        books_filter=args.books,
        testaments_filter=args.testaments,
        delay=args.delay,
    )

    chunks_path, full_path = save_outputs(all_chunks, full_data, args.output_dir)

    if args.load_chromadb:
        load_into_chromadb(str(chunks_path))