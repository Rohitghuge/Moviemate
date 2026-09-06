"""
Fetch 1,000 highly rated, child-friendly movies from TMDB:
500 Hindi, 250 English, and 250 Marathi.
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

LANGUAGE_TARGETS = {"hi": 500, "en": 250, "mr": 250}
LANGUAGE_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

NUM_PAGES = 100
MAX_VIEWING_AGE = 12
MIN_VOTE_AVERAGE = 6.5
MIN_VOTE_COUNT = 100

CATEGORIES = [
    {"name": "all_time_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 7.0, "vote_count.gte": 300},
    {"name": "blockbuster", "sort_by": "popularity.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 500},
    {"name": "new", "sort_by": "primary_release_date.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": MIN_VOTE_COUNT, "primary_release_date.lte": "2026-08-23"},
    {"name": "all_time_blockbuster", "sort_by": "vote_count.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 1000},
]

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


def is_under_13(certification):
    """Keep only certifications that explicitly allow viewers under 13."""
    cert = (certification or "").upper().replace(" ", "")
    return cert in {"G", "PG", "U", "UA", "U/A", "7", "7A", "12", "12A", "TV-G", "TV-PG"}


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
    """Fetch highly rated, under-13 movies for one exact language quota."""
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
                certification = get_certification(movie_id)
                if not is_under_13(certification):
                    continue

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
                    "certification": certification,
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

    for language_code, limit in LANGUAGE_TARGETS.items():
        fetch_for_language(language_code, limit, CATEGORIES, genre_map, all_movies, seen_ids)

    counts = {code: sum(movie["original_language"] == code for movie in all_movies)
              for code in LANGUAGE_TARGETS}
    if len(all_movies) != sum(LANGUAGE_TARGETS.values()):
        raise SystemExit(
            "TMDB did not provide exactly 400 qualifying movies. "
            f"Got {len(all_movies)} with counts {counts}."
        )
    print("Final language counts:", counts)

    return all_movies


if __name__ == "__main__":
    movies = fetch_movies()
    save_progress(movies)
    print(f"\nDone! Saved {len(movies)} movies to {OUTPUT_FILE}")
