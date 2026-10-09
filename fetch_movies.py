
import json
import os
import time
import requests
from pathlib import Path

# ============================================================
# MovieMate Movie Database Builder
# Targets: 150 Marathi + 150 English + 100 Hindi = 400 movies
# Horror movies are collected first for each language.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "movies.json"
ENV_FILE = BASE_DIR / ".env"

API_BASE = "https://api.themoviedb.org/3"
TARGETS = {
    "mr": 150,
    "en": 150,
    "hi": 100,
}

LANGUAGE_NAMES = {
    "mr": "Marathi",
    "en": "English",
    "hi": "Hindi",
}

MAX_PAGES = 500
REQUEST_TIMEOUT = 25
REQUEST_DELAY = 0.05

# TMDB genre IDs
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

# Genres associated with each mood.
MOOD_GENRES = {
    "happy": [35, 10751, 16, 10402],
    "sad": [18, 10749],
    "action": [28, 12, 53],
    "romance": [10749, 18],
    "horror": [27, 53, 9648],
}

session = requests.Session()
API_KEY = None


# ------------------------------------------------------------
# Load API key from environment or .env
# ------------------------------------------------------------

def load_api_key():
    key = os.getenv("TMDB_API_KEY")

    if key:
        return key.strip().strip('"').strip("'")

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if line.startswith("TMDB_API_KEY="):
                value = line.split("=", 1)[1].strip()
                return value.strip('"').strip("'")

    return None


# ------------------------------------------------------------
# Make a TMDB API request
# ------------------------------------------------------------

def tmdb_get(endpoint, params=None):
    if not API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is missing. Add it to your .env file."
        )

    request_params = dict(params or {})
    request_params["api_key"] = API_KEY

    url = f"{API_BASE}{endpoint}"

    for attempt in range(3):
        try:
            response = session.get(
                url,
                params=request_params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                wait_time = int(
                    response.headers.get("Retry-After", 2)
                )
                time.sleep(max(wait_time, 2))
                continue

            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()

        except requests.RequestException as error:
            print(f"  API request failed: {error}")

            if attempt == 2:
                return {}

            time.sleep(2 * (attempt + 1))

    return {}


# ------------------------------------------------------------
# Save progress so collected movies are not lost
# ------------------------------------------------------------

def save_movies(movie_database):
    OUTPUT_FILE.write_text(
        json.dumps(
            movie_database,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# Convert TMDB genre IDs into names
# ------------------------------------------------------------

def get_genre_names(genre_ids):
    return [
        GENRES[genre_id]
        for genre_id in genre_ids
        if genre_id in GENRES
    ]


# ------------------------------------------------------------
# Assign moods using movie genres
# ------------------------------------------------------------

def get_moods(genre_ids):
    moods = []

    for mood, mood_genres in MOOD_GENRES.items():
        if any(genre in genre_ids for genre in mood_genres):
            moods.append(mood)

    # Every movie can also be recommended for surprise-me.
    moods.append("surprise_me")

    return list(dict.fromkeys(moods))


# ------------------------------------------------------------
# Create a valid TMDB discover query
# ------------------------------------------------------------

def build_discover_params(
    language,
    page,
    genre_ids=None,
    sort_by="popularity.desc",
):
    params = {
        "language": "en-US",
        "with_original_language": language,
        "sort_by": sort_by,
        "include_adult": "true",
        "include_video": "false",
        "page": page,
        "vote_count.gte": 0,
    }

    # TMDB expects a single with_genres parameter.
    # The | symbol means match ANY selected genre.
    if genre_ids:
        params["with_genres"] = "|".join(
            str(genre_id) for genre_id in genre_ids
        )

    return params


# ------------------------------------------------------------
# Fetch one discover page
# ------------------------------------------------------------

def fetch_discover_page(
    language,
    page,
    genre_ids=None,
    sort_by="popularity.desc",
):
    params = build_discover_params(
        language=language,
        page=page,
        genre_ids=genre_ids,
        sort_by=sort_by,
    )

    data = tmdb_get("/discover/movie", params)

    return data.get("results", [])


# ------------------------------------------------------------
# Collect movies for one language and one genre search
# ------------------------------------------------------------

def collect_category(
    language,
    database,
    target,
    genre_ids=None,
    category_name="General",
):
    page = 1
    added = 0

    while len(database) < target and page <= MAX_PAGES:
        print(
            f"  {LANGUAGE_NAMES[language]} | "
            f"{category_name} | Page {page} | "
            f"Total {len(database)}/{target}"
        )

        results = fetch_discover_page(
            language=language,
            page=page,
            genre_ids=genre_ids,
        )

        if not results:
            break

        for movie in results:
            movie_id = movie.get("id")

            if not movie_id:
                continue

            # Avoid duplicates.
            if movie_id in database:
                continue

            title = (
                movie.get("title")
                or movie.get("original_title")
                or ""
            ).strip()

            if not title:
                continue

            movie_genre_ids = movie.get("genre_ids", [])
            genre_names = get_genre_names(movie_genre_ids)

            record = {
                "id": movie_id,
                "title": title,
                "original_title": movie.get("original_title", title),
                "overview": movie.get("overview", ""),
                "original_language": language,
                "language": LANGUAGE_NAMES[language],
                "release_date": movie.get("release_date", ""),
                "year": (
                    movie.get("release_date", "")[:4]
                    if movie.get("release_date")
                    else None
                ),
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "vote_average": movie.get("vote_average", 0),
                "vote_count": movie.get("vote_count", 0),
                "popularity": movie.get("popularity", 0),
                "adult": movie.get("adult", False),
                "genres": genre_names,
                "genre_ids": movie_genre_ids,
                "moods": get_moods(movie_genre_ids),
                "age_rating": "NR",
                "certification": "NR",
                "providers": [],
                "source": "TMDB",
            }

            database[movie_id] = record
            added += 1

            if len(database) >= target:
                break

        # Stop if TMDB has no more pages.
        total_pages = min(
            data_page_count(results),
            MAX_PAGES,
        )

        if page >= total_pages:
            break

        page += 1

    return added


def data_page_count(results):
    # The result list itself does not include pagination metadata.
    # The caller uses a separate check below when needed.
    # Returning MAX_PAGES keeps pagination controlled by empty pages.
    return MAX_PAGES


# ------------------------------------------------------------
# Fetch discover results with pagination metadata
# ------------------------------------------------------------

def collect_category(
    language,
    database,
    target,
    genre_ids=None,
    category_name="General",
):
    page = 1
    added = 0

    while len(database) < target and page <= MAX_PAGES:
        print(
            f"  {LANGUAGE_NAMES[language]} | "
            f"{category_name} | Page {page} | "
            f"Total {len(database)}/{target}"
        )

        params = build_discover_params(
            language=language,
            page=page,
            genre_ids=genre_ids,
        )

        data = tmdb_get("/discover/movie", params)
        results = data.get("results", [])

        if not results:
            break

        for movie in results:
            movie_id = movie.get("id")

            if not movie_id or movie_id in database:
                continue

            title = (
                movie.get("title")
                or movie.get("original_title")
                or ""
            ).strip()

            if not title:
                continue

            genre_ids_for_movie = movie.get("genre_ids", [])
            release_date = movie.get("release_date", "")

            database[movie_id] = {
                "id": movie_id,
                "title": title,
                "original_title": movie.get("original_title", title),
                "overview": movie.get("overview", ""),
                "original_language": language,
                "language": LANGUAGE_NAMES[language],
                "release_date": release_date,
                "year": release_date[:4] if release_date else None,
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "vote_average": movie.get("vote_average", 0),
                "vote_count": movie.get("vote_count", 0),
                "popularity": movie.get("popularity", 0),
                "adult": movie.get("adult", False),
                "genres": get_genre_names(genre_ids_for_movie),
                "genre_ids": genre_ids_for_movie,
                "moods": get_moods(genre_ids_for_movie),
                "age_rating": "NR",
                "certification": "NR",
                "providers": [],
                "source": "TMDB",
            }

            added += 1

            if len(database) >= target:
                break

        save_movies_all_languages()

        total_pages = data.get("total_pages", 1)

        if page >= min(total_pages, MAX_PAGES):
            break

        page += 1

    return added


# ------------------------------------------------------------
# Shared database for saving progress
# ------------------------------------------------------------

ALL_MOVIES = {}


def save_movies_all_languages():
    save_movies(ALL_MOVIES)


# ------------------------------------------------------------
# Collect horror movies first, then other moods
# ------------------------------------------------------------

def fetch_for_language(language, target):
    print("\n" + "=" * 60)
    print(f"FETCHING {LANGUAGE_NAMES[language].upper()} MOVIES")
    print(f"Target: {target}")
    print("Horror movies will be prioritized first.")
    print("=" * 60)

    database = {}

    # Approximately 30% horror, where available.
    horror_target = max(1, int(target * 0.30))

    print(f"\n[1] Collecting {LANGUAGE_NAMES[language]} horror movies...")

    collect_category(
        language=language,
        database=database,
        target=horror_target,
        genre_ids=[27],
        category_name="Horror",
    )

    print(f"\nHorror collected: {len(database)}/{horror_target} target")

    # Fill the database using the requested mood categories.
    print("\n[2] Collecting movies for other moods...")

    for mood in ["action", "romance", "happy", "sad"]:
        if len(database) >= target:
            break

        print(f"\nSearching mood: {mood}")

        collect_category(
            language=language,
            database=database,
            target=target,
            genre_ids=MOOD_GENRES[mood],
            category_name=mood.title(),
        )

    # Final fallback: popular movies of the same language.
    if len(database) < target:
        print("\n[3] Filling remaining places with popular movies...")

        collect_category(
            language=language,
            database=database,
            target=target,
            genre_ids=None,
            category_name="Popular",
        )

    ALL_MOVIES.update(database)
    save_movies_all_languages()

    print(
        f"\nFinished {LANGUAGE_NAMES[language]}: "
        f"{len(database)}/{target} movies collected."
    )

    return database


# ------------------------------------------------------------
# Add certifications and streaming provider information
# ------------------------------------------------------------

def fetch_certification(movie_id):
    data = tmdb_get(f"/movie/{movie_id}/release_dates")

    results = data.get("results", [])

    # Prefer Indian certification, then US certification.
    for country_code in ["IN", "US", "GB"]:
        for country in results:
            if country.get("iso_3166_1") != country_code:
                continue

            for release in country.get("release_dates", []):
                certification = (
                    release.get("certification") or ""
                ).strip()

                if certification:
                    return certification

    return "NR"


def fetch_providers(movie_id):
    data = tmdb_get(f"/movie/{movie_id}/watch/providers")
    results = data.get("results", {})

    # India is the primary market for MovieMate.
    india = results.get("IN", {})
    provider_names = []

    for section in ["flatrate", "rent", "buy", "free", "ads"]:
        for provider in india.get(section, []):
            name = provider.get("provider_name")

            if name and name not in provider_names:
                provider_names.append(name)

    return provider_names


def enrich_movies():
    print("\n" + "=" * 60)
    print("ADDING CERTIFICATIONS AND STREAMING PROVIDERS")
    print("=" * 60)

    movies = list(ALL_MOVIES.values())

    for index, movie in enumerate(movies, start=1):
        movie_id = movie["id"]

        print(
            f"[{index}/{len(movies)}] "
            f"Updating {movie['title']}"
        )

        certification = fetch_certification(movie_id)

        movie["certification"] = certification
        movie["age_rating"] = certification

        movie["providers"] = fetch_providers(movie_id)

        save_movies_all_languages()

    print("Enrichment complete.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    global API_KEY

    print("=" * 60)
    print("MOVIEMATE MOVIE DATABASE BUILDER")
    print("=" * 60)
    print("Target: 400 unique movies")
    print("Marathi: 150 | English: 150 | Hindi: 100")
    print("Horror movies will be prioritized first.")
    print()

    API_KEY = load_api_key()

    if not API_KEY:
        print("ERROR: TMDB_API_KEY was not found.")
        print("Add this to your .env file:")
        print("TMDB_API_KEY=your_tmdb_api_key")
        return

    # Verify the API key before doing a large download.
    test = tmdb_get("/configuration")

    if not test:
        print("ERROR: TMDB API check failed.")
        print("Check your API key and internet connection.")
        return

    for language, target in TARGETS.items():
        fetch_for_language(language, target)

    save_movies_all_languages()

    print("\n" + "=" * 60)
    print("MOVIE COLLECTION SUMMARY")
    print("=" * 60)

    for language, target in TARGETS.items():
        count = sum(
            1
            for movie in ALL_MOVIES.values()
            if movie.get("original_language") == language
        )

        print(
            f"{LANGUAGE_NAMES[language]}: {count}/{target}"
        )

    print(f"Total collected: {len(ALL_MOVIES)}")
    print(f"Saved to: {OUTPUT_FILE}")

    # Optional: fetch certification and providers for every movie.
    # This makes many extra API requests and can take a while.
    enrich_choice = input(
        "\nFetch age certifications and streaming providers "
        "for every movie? (y/n): "
    ).strip().lower()

    if enrich_choice == "y":
        enrich_movies()

    save_movies_all_languages()

    print("\nDONE!")
    print(f"Movie database saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        save_movies_all_languages()
        print("\nStopped by user. Progress has been saved.")
    except Exception as error:
        save_movies_all_languages()
        print(f"\nERROR: {error}")
        print("Any collected movies have been saved.")