"""
build_index.py
Reads movies.json, converts each movie's overview into a vector embedding
using a local AI model, and stores everything in a ChromaDB database
(a local folder called chroma_db) so it can be searched by meaning later.

Run this once after fetch_movies.py has finished. If you ever change or
re-fetch movies.json, just run this again to rebuild the index.
"""

import json
from sentence_transformers import SentenceTransformer
import chromadb

INPUT_FILE = "movies.json"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "movies"
BATCH_SIZE = 50

print("Loading movies.json...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies")

# TMDB's "popular" list can shift while paginating, so the same movie
# sometimes appears on more than one page. Remove duplicates by ID.
seen_ids = set()
unique_movies = []
for m in movies:
    if m["id"] not in seen_ids:
        seen_ids.add(m["id"])
        unique_movies.append(m)
duplicates_removed = len(movies) - len(unique_movies)
movies = unique_movies
if duplicates_removed:
    print(f"Removed {duplicates_removed} duplicate movie(s). {len(movies)} unique movies remain.")

print("Loading embedding model (first run downloads ~80MB, please wait)...")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
# Start clean each time this script runs, so re-running never duplicates entries
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.create_collection(name=COLLECTION_NAME)

print("Generating embeddings and storing them...")
for i in range(0, len(movies), BATCH_SIZE):
    batch = movies[i:i + BATCH_SIZE]

    ids = [str(m["id"]) for m in batch]
    documents = [
        f"{m.get('title', 'Unknown')}. {m.get('overview') or ''} "
        f"Genres: {', '.join(m.get('genres', []))}. "
        f"Original language: {m.get('original_language') or ''}."
        for m in batch
    ]
    metadatas = [{
        "title": m.get("title") or "Unknown",
        "original_language": m.get("original_language") or "",
        "genres": ", ".join(m.get("genres", [])),
        "certification": m.get("certification") or "NR",
        "watch_providers": ", ".join(m.get("watch_providers", [])),
        "release_date": m.get("release_date") or "",
        "rating": m.get("rating") or 0,
        "poster_path": m.get("poster_path") or "",
    } for m in batch]

    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  Indexed {min(i + BATCH_SIZE, len(movies))}/{len(movies)} movies")

print(f"\nDone! Your searchable movie index is ready in the '{CHROMA_PATH}' folder.")
