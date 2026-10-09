"""
Fetch 400 movies from TMDB across Hindi, Marathi and English — 150 Marathi,
150 English, 100 Hindi — covering ALL age ratings (U up to A/R, not just
under-13 content), with a guaranteed slice of each mood genre (Comedy,
Drama, Action, Romance, Horror) in every language, not just whatever the
general "most popular" lists happen to turn up.

Why this version is different from the old fetch_movies.py:
- The old script only kept movies certified for under-13 viewing
  (MAX_VIEWING_AGE = 12). Real horror movies are almost always certified
  above that, so the old catalog ended up with almost no Horror movies at
  all -- which is exactly why the "Horror" mood button kept coming back
  empty. This version keeps every certification (U, U/A 7+/13+/16+, A, PG,
  PG-13, R, NC-17, ...) so horror, thriller and other mature genres are
  actually represented. The app's own age filtering (get_min_age() in
  app.py) still hides age-inappropriate movies from a given user at
  recommend-time -- that's the right place for that check, not here.
- The old script fetched "popular"/"top-rated" lists per language with no
  genre awareness, so a mood like Horror or Romance was only as well
  represented as however many happened to show up in the generic blockbuster
  lists. This version reserves a per-language, per-genre quota (via TMDB's
  with_genres filter) for Comedy/happy, Drama/sad, Action, Romance and
  Horror specifically, on top of a general "all-time best / blockbuster"
  pool for variety -- so every mood has real, intentionally-fetched movies
  in every language, not leftovers.

Run this once (after setting TMDB_API_KEY in your .env), then run
build_index.py again afterwards to rebuild the search index from the new
movies.json before redeploying.
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

# 150 Marathi + 150 English + 100 Hindi = 400
LANGUAGE_TARGETS = {"mr": 150, "en": 150, "hi": 100}
LANGUAGE_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

NUM_PAGES = 100
MIN_VOTE_AVERAGE = 6.0
MIN_VOTE_COUNT = 100

# TMDB genre ids for the moods the app supports.
MOOD_GENRES = [
    ("Comedy", 35),    # happy
    ("Drama", 18),     # sad
    ("Action", 28),    # action
    ("Romance", 10749),  # romance
    ("Horror", 27),    # horror
]

# General "best of all time / blockbuster" pool -- same spirit as before,
# just no longer age-restricted.
GENERAL_CATEGORIES = [
    {"name": "all_time_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 7.0, "vote_count.gte": 300},
    {"name": "blockbuster", "sort_by": "popularity.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 500},
    {"name": "new", "sort_by": "primary_release_date.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": MIN_VOTE_COUNT, "primary_release_date.lte": "2026-08-23"},
    {"name": "all_time_blockbuster", "sort_by": "vote_count.desc", "vote_average.gte": MIN_VOTE_AVERAGE, "vote_count.gte": 1000},
]

# Marathi has far less catalogue depth on TMDB (especially for Horror), so
# both the general pool and the per-genre pools use much lower vote
# thresholds for it -- otherwise most quotas would simply come back empty.
GENERAL_CATEGORIES_REGIONAL = [
    {"name": "regional_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 6.0, "vote_count.gte": 5},
    {"name": "regional_popular", "sort_by": "popularity.desc", "vote_average.gte": 5.0, "vote_count.gte": 2},
    {"name": "regional_new", "sort_by": "primary_release_date.desc", "vote_average.gte": 5.0, "vote_count.gte": 1, "primary_release_date.lte": "2026-08-23"},
]

# Fraction of each language's quota spent on the general pool; the rest is
# split evenly across the 5 mood genres above.
GENERAL_SHARE = 0.4

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


def fetch_quota(language_code, limit, categories, label, genre_map, all_movies, seen_ids, extra_params=None):
    """Fetch up to `limit` new movies for one language from a set of
    /discover/movie category filters (general categories, or a single
    genre's filter via extra_params), skipping ids already collected."""
    start_count = len(all_movies)
    lang_name = LANGUAGE_NAMES.get(language_code, language_code)

    for category in categories:
        if len(all_movies) - start_count >= limit:
            break
        print(f"\nFetching {label} / {category['name']} ({lang_name})...")

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
            if language_code:
                params["with_original_language"] = language_code
            if extra_params:
                params.update(extra_params)

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
                # Every certification is kept -- U, U/A 7+/13+/16+, A, PG,
                # PG-13, R, NC-17, unrated, all of it. app.py's own
                # get_min_age() filtering handles hiding age-inappropriate
                # movies from a given user at recommend-time; restricting
                # the catalogue itself here is what starved the Horror mood
                # (and R-rated content generally) last time.
                certification = get_certification(movie_id)

                all_movies.append({
                    "id": movie_id,
                    "title": movie.get("title"),
                    "overview": movie.get("overview"),
                    "original_language": lang_code,
                    "original_language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
                    "categories": [category["name"]],
                    "mood_label": label,
                    "genres": genres,
                    "release_date": movie.get("release_date"),
                    "rating": movie.get("vote_average"),
                    "certification": certification,
                    "watch_providers": get_watch_providers(movie_id),
                    "poster_path": movie.get("poster_path"),
                })
                time.sleep(0.3)

            save_progress(all_movies)
            print(f"  Page {page}: {len(all_movies) - start_count}/{limit} {lang_name} ({label}) so far (saved)")

    got = len(all_movies) - start_count
    print(f"-> {lang_name} / {label}: got {got}/{limit}")
    return got


def fetch_for_language(language_code, limit, genre_map, all_movies, seen_ids):
    is_regional = language_code == "mr"
    general_categories = GENERAL_CATEGORIES_REGIONAL if is_regional else GENERAL_CATEGORIES

    general_limit = round(limit * GENERAL_SHARE)
    remaining = limit - general_limit
    per_genre_limit = remaining // len(MOOD_GENRES)
    leftover = remaining - per_genre_limit * len(MOOD_GENRES)

    fetch_quota(language_code, general_limit, general_categories, "general", genre_map, all_movies, seen_ids)

    for i, (genre_name, genre_id) in enumerate(MOOD_GENRES):
        quota = per_genre_limit + (1 if i < leftover else 0)
        # Genre-specific discover query: same sort/vote-quality filters as
        # the general pool for this language, plus with_genres to guarantee
        # the mood is actually represented.
        categories = general_categories if is_regional else GENERAL_CATEGORIES
        fetch_quota(
            language_code, quota, categories, genre_name, genre_map, all_movies, seen_ids,
            extra_params={"with_genres": genre_id},
        )

    # Top off with an unrestricted pass (any remaining shortfall, any genre)
    # so a language still hits its overall target even if one mood came up
    # short on TMDB.
    got_so_far = sum(
        1 for m in all_movies
        if m["original_language"] == language_code
    )
    shortfall = limit - got_so_far
    if shortfall > 0:
        print(f"\n{LANGUAGE_NAMES.get(language_code, language_code)} short by {shortfall}, topping off with a general pass...")
        fetch_quota(language_code, shortfall, general_categories, "topoff", genre_map, all_movies, seen_ids)


def fetch_movies():
    print("Fetching genre list...")
    genre_map = get_genre_map()

    all_movies = []
    seen_ids = set()

    for language_code, limit in LANGUAGE_TARGETS.items():
        fetch_for_language(language_code, limit, genre_map, all_movies, seen_ids)

    counts = {code: sum(movie["original_language"] == code for movie in all_movies)
              for code in LANGUAGE_TARGETS}
    expected_total = sum(LANGUAGE_TARGETS.values())
    if len(all_movies) != expected_total or counts != LANGUAGE_TARGETS:
        print(
            f"\nWarning: didn't hit the exact target of {expected_total} movies "
            f"(got {len(all_movies)}, counts {counts}). This can happen if TMDB "
            f"simply doesn't have enough qualifying titles for one language/genre "
            f"combo (Marathi Horror in particular is a thin category on TMDB). "
            f"Everything found so far has still been saved to {OUTPUT_FILE}."
        )
    else:
        print("Final language counts:", counts)

    mood_counts = {}
    for m in all_movies:
        mood_counts.setdefault(m["original_language"], {}).setdefault(m.get("mood_label", "general"), 0)
        mood_counts[m["original_language"]][m.get("mood_label", "general")] += 1
    print("\nMood/genre breakdown per language:")
    for lang, moods in mood_counts.items():
        print(f"  {LANGUAGE_NAMES.get(lang, lang)}: {moods}")

    return all_movies


if __name__ == "__main__":
    movies = fetch_movies()
    save_progress(movies)
    print(f"\nDone! Saved {len(movies)} movies to {OUTPUT_FILE}")
