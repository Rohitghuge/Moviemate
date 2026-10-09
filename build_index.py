"""
build_index.py
Reads movies.json and builds the TF-IDF search index (movie_index.pkl) that
app.py loads at startup for retrieve_movies(). Run this every time
movies.json changes (e.g. after running fetch_movies.py again), then
redeploy so the live app picks up the new movies.json + movie_index.pkl
together.
"""

import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "movies.json"
OUTPUT_FILE = "movie_index.pkl"

print("Loading movies.json...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies")

# De-duplicate by id, same as before (TMDB's lists can repeat a movie across
# pages/categories).
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


def genres_text(m):
    genres = m.get("genres", "")
    if isinstance(genres, list):
        return ", ".join(genres)
    return genres or ""


def corpus_text(m):
    return (
        f"{m.get('title', '')}. {m.get('overview') or ''} "
        f"Genres: {genres_text(m)}. "
        f"Language: {m.get('original_language_name') or m.get('original_language') or ''}."
    )


ids = [str(m["id"]) for m in movies]
documents = [corpus_text(m) for m in movies]

print("Building TF-IDF index...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
)
matrix = vectorizer.fit_transform(documents)
print(f"TF-IDF matrix shape: {matrix.shape}")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "ids": ids}, f)

print(f"\nDone! Saved index for {len(ids)} movies to {OUTPUT_FILE}")
