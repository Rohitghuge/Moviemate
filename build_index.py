"""
build_index.py
Reads movies.json and builds a TF-IDF search index over each movie's title +
overview + genres. TF-IDF is a lightweight, classic text-similarity technique
(no neural network, no onnxruntime, no GPU) that works well for a small,
fixed catalog like this one (a few hundred movies).

This replaces the old fastembed + chromadb approach, which needed
onnxruntime (100-300MB+ of RAM just for the runtime) — overkill for 300
movies and the main cause of out-of-memory crashes on small hosting plans
(e.g. Render's free 512MB tier).

Run this once after fetch_movies.py has finished. If you ever change or
re-fetch movies.json, just run this again to rebuild the index.

Output: movie_index.pkl — a single small file (usually a few hundred KB)
containing the fitted vectorizer, the TF-IDF matrix, and the row->movie_id
mapping. app.py loads this file directly; there is no separate database
folder to manage or keep persistent on disk.
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

print("Building TF-IDF index...")
ids = [str(m["id"]) for m in movies]
documents = [
    f"{m.get('title', 'Unknown')}. {m.get('overview') or ''} "
    # genre words repeated so they carry real weight against the overview
    # text — otherwise a short tag like "Horror" gets drowned out by a
    # long plot summary that never uses the word itself.
    f"Genres: {(', '.join(m.get('genres', [])) + ' ') * 3}. "
    f"Original language: {m.get('original_language') or ''}."
    for m in movies
]

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# Generic words people type in chat ("suggest a romantic movie") shouldn't
# count as a real match just because a title happens to contain "Movie".
extra_stop_words = ENGLISH_STOP_WORDS | {"movie", "movies", "film", "films"}
vectorizer = TfidfVectorizer(stop_words=list(extra_stop_words), max_features=20000)
matrix = vectorizer.fit_transform(documents)  # sparse, tiny in memory

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "ids": ids}, f)

print(f"\nDone! Your searchable movie index is saved to '{OUTPUT_FILE}' "
      f"({matrix.shape[0]} movies x {matrix.shape[1]} terms).")
