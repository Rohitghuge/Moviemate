"""
fetch_movies.py
Fetches ~1000 well-known/highly-rated movies from TMDB across Hindi,
Marathi, and English, including genre, age certification, and streaming
platform availability. Saves everything into movies.json.

Strategy: Hindi and Marathi each get a realistic target (there simply aren't
1000s of highly-rated regional titles on TMDB). Whatever is short of 1000
after that gets filled with English movies, so you always end up at (close
to) your target count.
"""

import requests
import json
import time
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_FILE = "movies.json"

TARGET_MOVIES = 1000
LANGUAGE_TARGETS = {"hi": 250, "mr": 100}  # English fills everything else, up to TARGET_MOVIES
LANGUAGE_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

NUM_PAGES = 40
MIN_VOTE_AVERAGE = 6.0
MIN_VOTE_COUNT = 50

CATEGORIES = [
    {"name": "all_time_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 7.0, "vote_count.gte": 300},
    {"name": "blockbuster", "sort_by": "popularity.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 500},
    {"name": "new", "sort_by": "primary_release_date.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": MIN_VOTE_COUNT, "primary_release_date.lte": "2026-08-23"},
    {"name": "all_time_blockbuster", "sort_by": "vote_count.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 1000},
]

# Regional languages (Hindi, Marathi) have far fewer TMDB voters than
# Hollywood, so demanding 300-1000+ votes filters out almost everything.
# These looser thresholds are used only for hi/mr.
REGIONAL_CATEGORIES = [
    {"name": "regional_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 6.0, "vote_count.gte": 20},
    {"name": "regional_popular", "sort_by": "popularity.desc", "vote_average.gte": 5.0, "vote_count.gte": 10},
    {"name": "regional_new", "sort_by": "primary_release_date.desc", "vote_average.gte": 5.0, "vote_count.gte": 5, "primary_release_date.lte": "2026-08-23"},
]
# Fallback category used only to top off the count if the above aren't enough
TOPOFF_CATEGORY = {"name": "popular_fill", "sort_by": "popularity.desc", "vote_average.gte": 5.0, "vote_count.gte": 20}

if not TMDB_API_KEY:
    raise SystemExit(
        "TMDB_API_KEY not found. Make sure your .env file exists in this "
        "same folder and has a line like: TMDB_API_KEY=your_key_here"
    )

session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=1.5,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


def safe_get(url, params, timeout=15):
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: request failed after retries ({e}). Skipping this one.")
        return None


def get_genre_map():
    data = safe_get(f"{BASE_URL}/genre/movie/list", {"api_key": TMDB_API_KEY, "language": "en-US"})
    return {g["id"]: g["name"] for g in data["genres"]} if data else {}


def get_certification(movie_id):
    data = safe_get(f"{BASE_URL}/movie/{movie_id}/release_dates", {"api_key": TMDB_API_KEY})
    if not data:
        return "NR"
    for country_code in ["IN", "US"]:
        for entry in data.get("results", []):
            if entry["iso_3166_1"] == country_code:
                for rd in entry["release_dates"]:
                    if rd.get("certification"):
                        return rd["certification"]
    return "NR"


def get_watch_providers(movie_id):
    data = safe_get(f"{BASE_URL}/movie/{movie_id}/watch/providers", {"api_key": TMDB_API_KEY})
    if not data:
        return []
    region_data = data.get("results", {}).get("IN", {})
    providers = []
    for category in ["flatrate", "free", "ads"]:
        for p in region_data.get(category, []):
            providers.append(p["provider_name"])
    return sorted(set(providers))


def save_progress(movies):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)


def fetch_for_language(language_code, limit, categories, genre_map, all_movies, seen_ids):
    """Fetch movies for one language until `limit` new movies are added or supply runs out."""
    start_count = len(all_movies)
    lang_name = LANGUAGE_NAMES.get(language_code, language_code)

    for category in categories:
        if len(all_movies) - start_count >= limit:
            break
        print(f"\nFetching {category['name']} ({lang_name})...")

        for page in range(1, NUM_PAGES + 1):
            if len(all_movies) - start_count >= limit:
                break

            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "sort_by": category["sort_by"],
                **{k: v for k, v in category.items() if k != "name"},
                "page": page,
            }
            if language_code:  # empty/None means "any language" for the topoff pass
                params["with_original_language"] = language_code

            data = safe_get(f"{BASE_URL}/discover/movie", params)
            if not data or not data.get("results"):
                break  # no more pages / no more supply for this filter combo

            for movie in data["results"]:
                if len(all_movies) - start_count >= limit:
                    break
                movie_id = movie["id"]
                if movie_id in seen_ids:
                    continue
                seen_ids.add(movie_id)

                genres = [genre_map.get(gid, "") for gid in movie.get("genre_ids", [])]
                lang_code = movie.get("original_language")
                all_movies.append({
                    "id": movie_id,
                    "title": movie.get("title"),
                    "overview": movie.get("overview"),
                    "original_language": lang_code,
                    "original_language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
                    "categories": [category["name"]],
                    "genres": genres,
                    "release_date": movie.get("release_date"),
                    "rating": movie.get("vote_average"),
                    "certification": get_certification(movie_id),
                    "watch_providers": get_watch_providers(movie_id),
                    "poster_path": movie.get("poster_path"),
                })
                time.sleep(0.3)

            save_progress(all_movies)
            print(f"  Page {page}: {len(all_movies) - start_count}/{limit} {lang_name} movies so far (saved)")

    got = len(all_movies) - start_count
    print(f"-> {lang_name}: got {got}/{limit}")
    return got


def fetch_movies():
    print("Fetching genre list...")
    genre_map = get_genre_map()

    all_movies = []
    seen_ids = set()

    # Step 1: Hindi and Marathi, each up to their own realistic target,
    # using looser quality thresholds since TMDB has far fewer regional voters
    for language_code, limit in LANGUAGE_TARGETS.items():
        fetch_for_language(language_code, limit, REGIONAL_CATEGORIES, genre_map, all_movies, seen_ids)

    # Step 2: English fills the rest, using the same quality categories
    remaining = TARGET_MOVIES - len(all_movies)
    if remaining > 0:
        print(f"\nFilling remaining {remaining} slots with English movies...")
        fetch_for_language("en", remaining, CATEGORIES, genre_map, all_movies, seen_ids)

    # Step 3: Safety net — if still short (rare), top off with a looser filter, any language
    remaining = TARGET_MOVIES - len(all_movies)
    if remaining > 0:
        print(f"\nStill {remaining} short of target, topping off with a wider search...")
        fetch_for_language(None, remaining, [TOPOFF_CATEGORY], genre_map, all_movies, seen_ids)

    return all_movies


if __name__ == "__main__":
    movies = fetch_movies()
    save_progress(movies)
    print(f"\nDone! Saved {len(movies)} movies to {OUTPUT_FILE}")
