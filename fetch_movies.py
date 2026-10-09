"""
Fetch 400 movies from TMDB across Hindi, Marathi and English — 150 Marathi,
150 English, 100 Hindi — covering ALL age ratings (U up to A/R, not just
under-13 content), with a guaranteed slice of each mood genre (Comedy,
Drama, Action, Romance, Horror) in every language, not just whatever the
general "most popular" lists happen to turn up.

Changes from the previous version (why Horror kept coming back as only 1-2
movies even though this script was supposed to reserve ~12-18 per
language):

- The per-genre fetch was reusing the SAME strict vote_average/vote_count
  thresholds as the "all-time best of the best" general pool (e.g.
  vote_average >= 7.0, vote_count >= 300-1000). Horror movies are
  consistently rated lower on average than mainstream drama/comedy hits
  and have smaller vote counts, so almost none cleared that bar -- the
  genre-specific TMDB query for Horror was coming back nearly empty. This
  version uses its own, much looser tiers of thresholds for genre-specific
  fetching (and tries progressively looser tiers if the first comes up
  short), since the goal here is realistic mood coverage, not "only the
  most acclaimed movies of all time."
- The old "topoff" pass (used when a language fell short of its overall
  400-movie target) was NOT genre-aware: it just pulled in more movies of
  ANY genre to hit the language total, which silently diluted Horror's
  share right back down even if the per-genre fetch above had been fixed.
  This version tops off each genre's OWN shortfall first (so Horror
  specifically gets topped up with more Horror, not Comedy), and only
  falls back to an any-genre topoff for whatever gap is left after that.

Run this once (after setting TMDB_API_KEY in your .env), then run
build_index.py again afterwards to rebuild the search index from the new
movies.json before redeploying. The script prints a mood/language
breakdown at the end -- check that Horror's numbers look right before you
redeploy.
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
    {"name": "blockbuster", "sort_by": "popularity.desc", "vote_average.gte": 6.0, "vote_count.gte": 500},
    {"name": "new", "sort_by": "primary_release_date.desc", "vote_average.gte": 6.0, "vote_count.gte": 100, "primary_release_date.lte": "2026-08-23"},
    {"name": "all_time_blockbuster", "sort_by": "vote_count.desc", "vote_average.gte": 6.0, "vote_count.gte": 1000},
]

GENERAL_CATEGORIES_REGIONAL = [
    {"name": "regional_favorite", "sort_by": "vote_average.desc", "vote_average.gte": 6.0, "vote_count.gte": 5},
    {"name": "regional_popular", "sort_by": "popularity.desc", "vote_average.gte": 5.0, "vote_count.gte": 2},
    {"name": "regional_new", "sort_by": "primary_release_date.desc", "vote_average.gte": 5.0, "vote_count.gte": 1, "primary_release_date.lte": "2026-08-23"},
]

# Progressively looser filter tiers for genre-specific fetching. Tried in
# order, each only to fill whatever quota the previous tier didn't reach.
# Popularity-sorted throughout so the best-known titles in each genre still
# come first within a tier.
MOOD_TIERS_MAIN = [
    {"sort_by": "popularity.desc", "vote_average.gte": 5.5, "vote_count.gte": 50},
    {"sort_by": "popularity.desc", "vote_average.gte": 4.0, "vote_count.gte": 10},
    {"sort_by": "popularity.desc", "vote_count.gte": 1},
]
MOOD_TIERS_REGIONAL = [
    {"sort_by": "popularity.desc", "vote_average.gte": 4.5, "vote_count.gte": 5},
    {"sort_by": "popularity.desc", "vote_count.gte": 1},
    {"sort_by": "popularity.desc"},
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


def fetch_quota(language_code, limit, filters_list, label, genre_map, all_movies, seen_ids, extra_params=None):
    """Fetch up to `limit` NEW movies for one language by trying each filter
    set in `filters_list` in order, moving to the next only once a filter
    set runs out of pages/results, stopping as soon as the limit is hit."""
    start_count = len(all_movies)
    lang_name = LANGUAGE_NAMES.get(language_code, language_code)

    for filters in filters_list:
        if len(all_movies) - start_count >= limit:
            break
        tier_desc = ", ".join(f"{k}={v}" for k, v in filters.items() if k != "sort_by")
        print(f"\nFetching {label} ({lang_name}) -- {tier_desc or 'no quality filter'}...")

        for page in range(1, NUM_PAGES + 1):
            if len(all_movies) - start_count >= limit:
                break

            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page,
                **filters,
            }
            if language_code:
                params["with_original_language"] = language_code
            if extra_params:
                params.update(extra_params)

            data = safe_get(f"{BASE_URL}/discover/movie", params)
            if not data or not data.get("results"):
                break  # no more pages / no more supply for this filter tier

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
                # movies from a given user at recommend-time.
                certification = get_certification(movie_id)

                all_movies.append({
                    "id": movie_id,
                    "title": movie.get("title"),
                    "overview": movie.get("overview"),
                    "original_language": lang_code,
                    "original_language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
                    "categories": [label],
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
    mood_tiers = MOOD_TIERS_REGIONAL if is_regional else MOOD_TIERS_MAIN

    general_limit = round(limit * GENERAL_SHARE)
    remaining = limit - general_limit
    per_genre_limit = remaining // len(MOOD_GENRES)
    leftover = remaining - per_genre_limit * len(MOOD_GENRES)

    fetch_quota(language_code, general_limit, general_categories, "general", genre_map, all_movies, seen_ids)

    genre_quotas = {}
    for i, (genre_name, genre_id) in enumerate(MOOD_GENRES):
        genre_quotas[genre_name] = per_genre_limit + (1 if i < leftover else 0)
        fetch_quota(
            language_code, genre_quotas[genre_name], mood_tiers, genre_name, genre_map, all_movies, seen_ids,
            extra_params={"with_genres": MOOD_GENRES[i][1]},
        )

    # If a specific genre still fell short of ITS quota (e.g. Marathi
    # Horror just doesn't have enough titles on TMDB even at the loosest
    # tier), make one more dedicated attempt for that same genre with no
    # quality filter at all before giving up on it -- better to fill the
    # mood with lower-popularity titles than to quietly replace it with an
    # unrelated genre.
    for genre_name, genre_id in MOOD_GENRES:
        got = sum(1 for m in all_movies if m.get("mood_label") == genre_name and m["original_language"] == language_code)
        short = genre_quotas[genre_name] - got
        if short > 0:
            fetch_quota(
                language_code, short, [{"sort_by": "popularity.desc"}], genre_name, genre_map, all_movies, seen_ids,
                extra_params={"with_genres": genre_id},
            )

    # Only now, as an absolute last resort, top off any remaining overall
    # shortfall with any genre -- this no longer eats into the per-genre
    # quotas above since those were already retried on their own.
    got_so_far = sum(1 for m in all_movies if m["original_language"] == language_code)
    shortfall = limit - got_so_far
    if shortfall > 0:
        print(f"\n{LANGUAGE_NAMES.get(language_code, language_code)} still short by {shortfall} after genre retries, topping off with a general pass...")
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
    print("\nMood/genre breakdown per language (this is what to check before redeploying):")
    for lang, moods in mood_counts.items():
        print(f"  {LANGUAGE_NAMES.get(lang, lang)}: {moods}")
    horror_total = sum(moods.get("Horror", 0) for moods in mood_counts.values())
    print(f"\nTotal Horror movies across all languages: {horror_total}")

    return all_movies


if __name__ == "__main__":
    movies = fetch_movies()
    save_progress(movies)
    print(f"\nDone! Saved {len(movies)} movies to {OUTPUT_FILE}")
