"""
build_chroma.py
===============
1. Unzips a zip file containing JSON chunk files from your partner
2. Concatenates all chunks into a single list
3. Builds ChromaDB from the combined data

Usage:
    python build_chroma.py --zip genesis_chunks.zip
    python build_chroma.py --zip genesis_chunks.zip --chroma-path ../chroma
"""

import json
import zipfile
import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# ── Config ───────────────────────────────────────────────────────
COLLECTION_NAME = "genz_bible"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


# ── Step 1: Extract and concatenate chunks ───────────────────────
def load_chunks_from_zip(zip_path: str) -> list:
    """
    Unzips the file and concatenates all .json chunk files found inside.
    Handles both a flat zip and a zip with nested folders.
    """
    zip_path = Path(zip_path)
    all_chunks = []

    print(f"📦 Opening zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        json_files = [f for f in z.namelist() if f.endswith(".json")]
        print(f"   Found {len(json_files)} JSON files inside")

        for json_file in sorted(json_files):  # sorted for consistent ordering
            with z.open(json_file) as f:
                data = json.load(f)
                # handle both a list of chunks or a nested dict structure
                if isinstance(data, list):
                    all_chunks.extend(data)
                elif isinstance(data, dict):
                    # e.g. {"Genesis": {"chapters": {"1": [...chunks...]}}}
                    for book, book_data in data.items():
                        if isinstance(book_data, dict) and "chapters" in book_data:
                            for chapter_chunks in book_data["chapters"].values():
                                all_chunks.extend(chapter_chunks)
                        elif isinstance(book_data, list):
                            all_chunks.extend(book_data)

            print(f"   ✅ Loaded {json_file} ({len(all_chunks)} chunks total so far)")

    print(f"\n✅ Total chunks loaded: {len(all_chunks)}")
    return all_chunks


# ── Step 2: Prepare documents for ChromaDB ───────────────────────
def prepare_documents(chunks: list) -> dict:
    """
    Prepares documents, metadatas, and ids from the combined chunks.
    - Skips malformed chunks (verse == 0)
    - Deduplicates by id
    - Indexes KJV text for search, stores genz_text in metadata
    """
    documents = []
    metadatas = []
    ids = []
    seen_ids = set()
    skipped = 0

    for c in chunks:
        # Skip malformed chunks
        if c.get("verse") == 0:
            skipped += 1
            continue

        # Deduplicate
        chunk_id = c.get("id")
        if not chunk_id or chunk_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(chunk_id)

        documents.append(c.get("text", ""))
        metadatas.append({
            "reference": c.get("reference", ""),
            "book":      c.get("book", ""),
            "chapter":   str(c.get("chapter", "")),
            "verse":     str(c.get("verse", "")),
            "testament": c.get("testament") or "",
            "genz_text": c.get("genz_text", ""),  # empty string if not yet scraped
        })
        ids.append(chunk_id)

    print(f"   Prepared: {len(documents)} chunks  |  Skipped: {skipped}")
    return {"documents": documents, "metadatas": metadatas, "ids": ids}


# ── Step 3: Build ChromaDB ────────────────────────────────────────
def build_chromadb(chunks: list, chroma_path: str):
    """
    Creates (or recreates) the ChromaDB collection from the given chunks.
    """
    print(f"\n🔧 Loading embedding model: {EMBEDDING_MODEL}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    print("✅ Embedding model ready")

    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "GenZ Bible translation for RAG-powered chatbot"},
    )

    data = prepare_documents(chunks)

    # Upsert in batches
    batch_size = 500
    total_batches = (len(data["documents"]) + batch_size - 1) // batch_size
    print(f"\n📥 Loading {len(data['documents'])} chunks in {total_batches} batches...")

    for i in range(0, len(data["documents"]), batch_size):
        batch_num = i // batch_size + 1
        collection.add(
            documents=data["documents"][i:i + batch_size],
            metadatas=data["metadatas"][i:i + batch_size],
            ids=data["ids"][i:i + batch_size],
        )
        print(f"  ✅ Batch {batch_num}/{total_batches} done")

    print(f"\n🎉 ChromaDB built! Collection '{COLLECTION_NAME}' has {collection.count()} verses")
    print(f"   Saved to: {chroma_path}")
    return collection


# ── Step 4: Quick sanity check ────────────────────────────────────
def sanity_check(collection):
    print("\n🔍 Running sanity check queries...")
    test_queries = [
        "God created the heavens and the earth",
        "Adam and Eve in the garden",
        "Noah and the flood",
    ]
    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=2)
        print(f"\n  Q: {query}")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            genz = meta.get("genz_text", "")
            print(f"    → {meta['reference']}")
            print(f"       KJV:  {doc[:80]}")
            if genz:
                print(f"       GenZ: {genz[:80]}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB from zipped chunk files")
    parser.add_argument("--zip",        default="../genz_bible_data/genesis.zip", required=False,        help="Path to the zip file from your partner")
    parser.add_argument("--chroma-path", default="../chroma", help="Where to save ChromaDB (default: ../chroma)")
    args = parser.parse_args()

    # Run pipeline
    chunks     = load_chunks_from_zip(args.zip)
    collection = build_chromadb(chunks, args.chroma_path)
    sanity_check(collection)

    print("\n✅ All done! You can now commit the chroma/ folder to GitHub.")
    print(f"   git add {args.chroma_path}")
    print(f"   git commit -m 'add chromadb with genesis data'")