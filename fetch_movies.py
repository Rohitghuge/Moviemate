
import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
OUTPUT_FILE = "movies.json"

TARGETS = {
    "mr": 150,
    "hi": 100,
    "en": 100,
}

LANGUAGE_NAMES = {
    "mr": "Marathi",
    "hi": "Hindi",
    "en": "English",
}

MOOD_GENRES = {
    "happy": [35, 10751, 16, 10402],
    "sad": [18, 10749],
    "action": [28, 12, 53],
    "romance": [10749, 18],
    "horror": [27, 53, 9648],
}

session = requests.Session()
session.headers.update({"Accept": "application/json"})


def tmdb_get(endpoint, params=None):
    """Request TMDB with retries for temporary connection failures."""
    if not API_KEY:
        raise RuntimeError("TMDB_API_KEY is missing from your .env file.")

    request_params = dict(params or {})
    request_params["api_key"] = API_KEY

    for attempt in range(5):
        try:
            response = session.get(
                f"{BASE_URL}/{endpoint.lstrip('/')}",
                params=request_params,
                timeout=(10, 45),
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 4:
                    wait = min(2 ** attempt, 20)
                    print(
                        f"TMDB returned {response.status_code}. "
                        f"Retrying in {wait} seconds..."
                    )
                    time.sleep(wait)
                    continue

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as error:
            if attempt == 4:
                raise RuntimeError(
                    f"TMDB connection failed after 5 attempts: {error}"
                ) from error

            wait = min(2 ** attempt, 20)
            print(
                f"Connection failed (attempt {attempt + 1}/5). "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)

    raise RuntimeError("TMDB request failed.")


def check_api():
    print("Checking TMDB API...")
    tmdb_get("configuration")
    print("TMDB API check successful!")


def load_existing_movies():
    if not os.path.exists(OUTPUT_FILE):
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return {
                str(key): value
                for key, value in data.items()
                if isinstance(value, dict) and value.get("title")
            }

        if isinstance(data, list):
            return {
                str(movie.get("id")): movie
                for movie in data
                if isinstance(movie, dict)
                and movie.get("id") is not None
                and movie.get("title")
            }

    except (json.JSONDecodeError, OSError) as error:
        print(f"Could not read existing movies.json: {error}")

    return {}


ALL_MOVIES = load_existing_movies()


def save_movies():
    temporary_file = OUTPUT_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            ALL_MOVIES,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporary_file, OUTPUT_FILE)


def normalize_movie(movie, language):
    movie_id = movie.get("id")
    title = movie.get("title") or movie.get("name")

    if movie_id is None or not title:
        return None

    return {
        "id": movie_id,
        "title": title,
        "original_title": movie.get("original_title", title),
        "overview": movie.get("overview", ""),
        "release_date": movie.get("release_date", ""),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
        "vote_average": movie.get("vote_average", 0),
        "rating": movie.get("vote_average", 0),
        "vote_count": movie.get("vote_count", 0),
        "original_language": movie.get("original_language", language),
        "original_language_name": LANGUAGE_NAMES[language],
        "language": LANGUAGE_NAMES[language],
        "genre_ids": movie.get("genre_ids", []),
        "popularity": movie.get("popularity", 0),
        "adult": movie.get("adult", False),
        "moods": [],
        "watch_providers": [],
        "providers": [],
        "certification": "",
    }


def collect_category(language, mood, genres, needed):
    added = 0
    page = 1
    seen_pages = set()

    while added < needed and page <= 500:
        params = {
            "language": "en-US",
            "region": "IN",
            "with_original_language": language,
            "with_genres": "|".join(map(str, genres)),
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "page": page,
        }

        data = tmdb_get("discover/movie", params)
        results = data.get("results", [])

        if not results:
            break

        for item in results:
            movie_id = str(item.get("id", ""))

            if not movie_id:
                continue

            if movie_id in ALL_MOVIES:
                movie = ALL_MOVIES[movie_id]
                moods = movie.setdefault("moods", [])
                if mood not in moods:
                    moods.append(mood)
                continue

            movie = normalize_movie(item, language)

            if movie is None:
                continue

            movie["moods"] = [mood]
            ALL_MOVIES[movie_id] = movie
            added += 1

            print(
                f"  {LANGUAGE_NAMES[language]} | {mood} | "
                f"{movie['title']} ({added}/{needed})"
            )

            if added >= needed:
                break

        save_movies()

        total_pages = data.get("total_pages", 1)
        if page >= total_pages or page in seen_pages:
            break

        seen_pages.add(page)
        page += 1

        time.sleep(0.25)

    return added


def fill_remaining(language, target):
    added = 0
    page = 1

    while len([
        movie for movie in ALL_MOVIES.values()
        if movie.get("original_language") == language
    ]) < target and page <= 500:

        params = {
            "language": "en-US",
            "region": "IN",
            "with_original_language": language,
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "page": page,
        }

        data = tmdb_get("discover/movie", params)
        results = data.get("results", [])

        if not results:
            break

        for item in results:
            movie_id = str(item.get("id", ""))

            if not movie_id or movie_id in ALL_MOVIES:
                continue

            movie = normalize_movie(item, language)

            if movie:
                movie["moods"] = ["popular"]
                ALL_MOVIES[movie_id] = movie
                added += 1

                print(
                    f"  {LANGUAGE_NAMES[language]} | popular | "
                    f"{movie['title']}"
                )

                save_movies()

                language_count = sum(
                    1 for existing in ALL_MOVIES.values()
                    if existing.get("original_language") == language
                )

                if language_count >= target:
                    return added

        if page >= data.get("total_pages", 1):
            break

        page += 1
        time.sleep(0.25)

    return added


def fetch_for_language(language, target):
    print(f"\n{'=' * 45}")
    print(f"Fetching {LANGUAGE_NAMES[language]} movies")
    print(f"Target: {target}")
    print(f"{'=' * 45}")

    existing_count = sum(
        1 for movie in ALL_MOVIES.values()
        if movie.get("original_language") == language
    )

    if existing_count >= target:
        print(f"Already have {existing_count} movies. Skipping.")
        return

    mood_quota = target // len(MOOD_GENRES)

    for mood, genres in MOOD_GENRES.items():
        existing_count = sum(
            1 for movie in ALL_MOVIES.values()
            if movie.get("original_language") == language
        )

        remaining_total = target - existing_count

        if remaining_total <= 0:
            break

        existing_mood_count = sum(
            1 for movie in ALL_MOVIES.values()
            if movie.get("original_language") == language
            and mood in movie.get("moods", [])
        )

        needed = min(
            max(0, mood_quota - existing_mood_count),
            remaining_total,
        )

        if needed:
            print(f"\nMood: {mood} | Looking for {needed} movies")
            collect_category(language, mood, genres, needed)

    existing_count = sum(
        1 for movie in ALL_MOVIES.values()
        if movie.get("original_language") == language
    )

    if existing_count < target:
        print("\nFilling remaining slots with popular movies...")
        fill_remaining(language, target)

    save_movies()


def main():
    if not API_KEY:
        print("ERROR: TMDB_API_KEY not found in .env")
        return

    try:
        check_api()

        for language, target in TARGETS.items():
            fetch_for_language(language, target)

        save_movies()

        print("\n" + "=" * 45)
        print("MOVIE FETCH SUMMARY")
        print("=" * 45)

        for language, target in TARGETS.items():
            count = sum(
                1 for movie in ALL_MOVIES.values()
                if movie.get("original_language") == language
            )

            print(
                f"{LANGUAGE_NAMES[language]}: {count}/{target}"
            )

        print(f"Total unique movies: {len(ALL_MOVIES)}")
        print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")

        if any(
            sum(
                1 for movie in ALL_MOVIES.values()
                if movie.get("original_language") == lang
            ) < target
            for lang, target in TARGETS.items()
        ):
            print(
                "\nWARNING: Some targets were not reached. "
                "TMDB may not have returned enough matching movies."
            )

    except (RuntimeError, requests.exceptions.RequestException) as error:
        save_movies()
        print(f"\nERROR: {error}")
        print("Progress saved. Fix the issue and run the script again.")


if __name__ == "__main__":
    main()