"""
app.py
The Flask web version of MovieMate, now with real accounts.

- Sign up / log in with username + password (age is captured once at signup)
- Every chat message is saved to Postgres, per user
- "Seen it" movies are saved to Postgres too, so they stay excluded even
  after logging out and back in, or after the server restarts
- Same RAG logic as before: mood search, age filtering, travel mode,
  "movies like X" similarity search, and "more info" follow-ups
"""

import os
import re
import json
import difflib
import uuid
import random
import requests
import psycopg2
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, request, jsonify, session, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pickle
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
HISTORY_LIMIT = 6

LIGHT_GENRES = {"Comedy", "Animation", "Adventure", "Family", "Music", "Fantasy"}
SIMILARITY_TRIGGERS = ["like ", "similar to", "similar", "such as", "in the style of", "in the vein of"]
INFO_TRIGGERS = ["info", "detail", "tell me more", "more about", "elaborate", "know more"]

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")
if not DATABASE_URL:
    raise SystemExit(
        "DATABASE_URL not found. Add a line like:\n"
        "DATABASE_URL=postgresql://user:password@host/dbname\n"
        "to your .env file (get this from Neon or Supabase)."
    )

# ---------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            age INTEGER NOT NULL,
            security_question TEXT,
            security_answer_hash TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    # In case this table already existed before security questions were added
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS security_question TEXT")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS security_answer_hash TEXT")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS watched_movies (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            movie_id TEXT NOT NULL,
            movie_title TEXT,
            watched_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, movie_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_movies (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            movie_id TEXT NOT NULL,
            movie_data JSONB NOT NULL,
            added_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, movie_id)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def get_excluded_ids(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT movie_id FROM watched_movies WHERE user_id = %s", (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {r["movie_id"] for r in rows}


def add_watched(user_id, movie_id, movie_title):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO watched_movies (user_id, movie_id, movie_title, watched_at)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (user_id, movie_id) DO NOTHING""",
        (user_id, movie_id, movie_title, datetime.utcnow()),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_recent_history(user_id, limit=HISTORY_LIMIT):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM chat_messages WHERE user_id = %s ORDER BY id DESC LIMIT %s",
        (user_id, limit),
    )
    rows = list(reversed(cur.fetchall()))
    cur.close()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(user_id, role, content):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
        (user_id, role, content, datetime.utcnow()),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_watchlist(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT movie_data FROM watchlist_movies WHERE user_id = %s ORDER BY added_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [json.loads(row["movie_data"]) if isinstance(row["movie_data"], str) else row["movie_data"] for row in rows]


def add_to_watchlist(user_id, movie):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO watchlist_movies (user_id, movie_id, movie_data, added_at)
           VALUES (%s, %s, %s::jsonb, %s)
           ON CONFLICT (user_id, movie_id) DO UPDATE SET movie_data = EXCLUDED.movie_data""",
        (user_id, str(movie["id"]), json.dumps(movie), datetime.utcnow()),
    )
    conn.commit()
    cur.close()
    conn.close()


def remove_from_watchlist(user_id, movie_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM watchlist_movies WHERE user_id = %s AND movie_id = %s",
        (user_id, str(movie_id)),
    )
    conn.commit()
    cur.close()
    conn.close()


init_db()

# ---------------------------------------------------------------------
# Simple in-memory rate limiting (no extra dependency needed)
# ---------------------------------------------------------------------

from collections import defaultdict
import time as _time

_request_log = defaultdict(list)
_pending_chat_requests = {}
RATE_LIMIT = 20        # max requests
RATE_WINDOW = 60       # per this many seconds


def is_rate_limited(key):
    now = _time.time()
    timestamps = _request_log[key]
    _request_log[key] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_request_log[key]) >= RATE_LIMIT:
        return True
    _request_log[key].append(now)
    return False

# ---------------------------------------------------------------------
# Movie data + search setup
# ---------------------------------------------------------------------

print("Loading movie list...")
with open(os.path.join(BASE_DIR, "movies.json"), "r", encoding="utf-8") as f:
    all_movies = json.load(f)
all_titles_lower = [m["title"].lower() for m in all_movies]
movies_by_id = {str(m.get("id")): m for m in all_movies}

print("Loading movie search index...")
with open(os.path.join(BASE_DIR, "movie_index.pkl"), "rb") as f:
    _index = pickle.load(f)
vectorizer = _index["vectorizer"]
tfidf_matrix = _index["matrix"]          # sparse, small in memory
index_ids = _index["ids"]                # row position -> movie id
id_to_row = {mid: i for i, mid in enumerate(index_ids)}
print(f"Ready! {len(index_ids)} movies loaded.")

app = Flask(__name__)
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise SystemExit(
        "SECRET_KEY not found. Add a line like SECRET_KEY=some_long_random_string to your .env file. "
        "This protects your login sessions — don't leave it hardcoded or guessable."
    )
app.secret_key = SECRET_KEY


def get_min_age(certification):
    cert = (certification or "").upper()
    if "18" in cert or cert == "A" or "NC-17" in cert:
        return 18
    if cert == "R" or "R-" in cert:
        return 17
    if "16" in cert:
        return 16
    if "13" in cert:
        return 13
    if "7" in cert:
        return 7
    return 0


def is_light_genre(movie):
    genres = movie.get("genres", "")
    genre_list = [g.strip() for g in genres.split(",")] if isinstance(genres, str) else genres
    return any(g in LIGHT_GENRES for g in genre_list)


def movie_genre_list(movie):
    genres = movie.get("genres", "")
    if isinstance(genres, str):
        return [g.strip() for g in genres.split(",") if g.strip()]
    return list(genres or [])


def movie_has_genre(movie, genre):
    return genre.lower() in {g.lower() for g in movie_genre_list(movie)}


# Mood-chip (and free-typed) queries like "Suggest a romantic movie" were
# being answered almost entirely off TF-IDF text similarity, with no actual
# check that the recommended movie's genre matched the mood asked for. Since
# words like "suggest", "movie", "feeling" are common across nearly every
# plot summary, the similarity score alone wasn't a reliable genre signal,
# so comedies/crime films (Dhamaal, Welcome, Andhadhun, Jolly LLB) were
# regularly turning up under "Romance". When a query clearly maps to one of
# these moods, we now restrict the candidate pool to movies actually tagged
# with that genre before ranking by similarity, instead of trusting
# similarity alone to find genre-appropriate movies.
MOOD_GENRE_TRIGGERS = (
    (("romantic", "romance", "love story", "date night"), "Romance"),
    (("horror", "scary", "spooky", "haunted", "creepy"), "Horror"),
    (("action movie", "action film", "intense action", "action-packed"), "Action"),
    (("feeling sad", "sad movie", "comforting"), "Drama"),
    (("feeling happy", "feel-good", "feel good", "feelgood", "cheerful",
      "lighthearted", "fun and feel"), "Comedy"),
)


def detect_target_genre(query):
    q = query.lower()
    for keywords, genre in MOOD_GENRE_TRIGGERS:
        if any(re.search(rf"\b{re.escape(k)}\b", q) for k in keywords):
            return genre
    return None


LOCATION_PROFILES = {
    "maharashtra": {
        "languages": {"mr", "marathi"},
        "keywords": ("maharashtra", "marathi", "mumbai", "pune", "kolhapur", "nashik", "nagpur"),
    },
    "goa": {"languages": set(), "keywords": ("goa", "goan", "panaji", "margao")},
    "delhi": {"languages": set(), "keywords": ("delhi", "new delhi", "dilli")},
}


def get_location_profile(location_state):
    location = re.sub(r"[^a-z]", "", (location_state or "").lower())
    for name, profile in LOCATION_PROFILES.items():
        if name in location:
            return profile
    if location_state:
        return {"languages": set(), "keywords": (location_state.lower(),)}
    return None


# Only nudge in local/regional picks when the user actually asks for them —
# not on every single mood/genre query. Without this check, a user whose
# detected location is Maharashtra would get Marathi/Mumbai-set movies
# force-inserted ahead of the real results for every query ("romance",
# "happy", "horror", ...), which is what caused Marathi titles like
# "Mumbai Pune Mumbai", "Pawankhind" and "Dhamaal" to show up everywhere
# regardless of the mood actually asked for.
LOCATION_TRIGGERS = (
    "regional", "local movie", "local film", "near me", "my state",
    "my city", "marathi", "maharashtrian", "set in mumbai", "set in pune",
    "set in goa", "set in delhi", "based in mumbai", "based in pune",
)


def is_location_query(query):
    q = query.lower()
    return any(trigger in q for trigger in LOCATION_TRIGGERS)


def location_match_score(movie, profile):
    if not profile:
        return 0
    source = {**movies_by_id.get(str(movie.get("id", "")), {}), **movie}
    title = (source.get("title") or "").lower()
    overview = (source.get("overview") or "").lower()
    language = (source.get("original_language") or "").lower()
    language_name = (source.get("original_language_name") or "").lower()
    score = 3 if language in profile["languages"] or language_name in profile["languages"] else 0
    for keyword in profile["keywords"]:
        pattern = rf"\b{re.escape(keyword)}\b"
        if re.search(pattern, title):
            score += 5
        elif re.search(pattern, overview):
            score += 2
    return score


def normalize_movie(m, movie_id=None):
    catalog_movie = movies_by_id.get(str(movie_id if movie_id is not None else m.get("id", "")), {})
    source = {**catalog_movie, **m}
    genres = source.get("genres", "")
    genres_str = ", ".join(genres) if isinstance(genres, list) else (genres or "")
    providers = source.get("watch_providers", "")
    providers_str = ", ".join(providers) if isinstance(providers, list) else (providers or "")
    return {
        "id": str(movie_id if movie_id is not None else source.get("id", "")),
        "title": source.get("title") or "Unknown",
        "overview": source.get("overview") or "",
        "original_language": source.get("original_language") or "",
        "original_language_name": source.get("original_language_name") or "",
        "genres": genres_str,
        "certification": source.get("certification") or "NR",
        "watch_providers": providers_str,
        "release_date": source.get("release_date") or "",
        "rating": source.get("rating") or 0,
        "poster_path": source.get("poster_path") or "",
    }


def find_title_matches(query, limit=2):
    query_lower = query.lower()
    matches = []
    for m, title_lower in zip(all_movies, all_titles_lower):
        if title_lower in query_lower or query_lower in title_lower:
            matches.append(m)
    if not matches:
        for word in query_lower.split():
            if len(word) < 4:
                continue
            close = difflib.get_close_matches(word, all_titles_lower, n=limit, cutoff=0.75)
            for c in close:
                idx = all_titles_lower.index(c)
                if all_movies[idx] not in matches:
                    matches.append(all_movies[idx])
    return matches[:limit]


def is_similarity_query(query):
    q = query.lower()
    return any(trigger in q for trigger in SIMILARITY_TRIGGERS)


def is_info_query(query):
    q = query.lower()
    return any(t in q for t in INFO_TRIGGERS)


def extract_number(query):
    match = re.search(r'\d+', query)
    return int(match.group()) if match else None


_trailer_cache = {}


def get_trailer_url(movie_id):
    if movie_id in _trailer_cache:
        return _trailer_cache[movie_id]
    url = None
    if TMDB_API_KEY:
        try:
            resp = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                params={"api_key": TMDB_API_KEY},
                timeout=8,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            trailer = next(
                (v for v in results if v.get("type") == "Trailer" and v.get("site") == "YouTube"),
                None,
            )
            if trailer:
                url = f"https://www.youtube.com/watch?v={trailer['key']}"
        except requests.exceptions.RequestException:
            url = None
    _trailer_cache[movie_id] = url
    return url


_live_providers_cache = {}


def get_live_providers(movie_id):
    """
    Fresh watch-provider info straight from TMDB, not the snapshot baked into
    movies.json at fetch time (which goes stale as availability changes —
    e.g. a movie added to Netflix after movies.json was built).

    Returns live provider names and TMDB's own JustWatch page link for this
    title/region. We deliberately do NOT try to scrape per-provider deep
    links out of that JustWatch page server-side: JustWatch's page content is
    rendered client-side by its own JavaScript, so a plain server-to-server
    HTTP fetch (requests.get) only ever receives the empty page shell, never
    the real "open in Netflix" / "open in Prime Video" links a browser would
    show after running that JavaScript. That approach was tried and always
    silently returned nothing, while still spending a network round trip on
    every single card.

    Instead, the frontend sends the user straight to each platform's own
    site (see PLATFORM_SEARCH_URLS in index.html), with that title's name
    pre-filled into the platform's own search. That is the honest ceiling
    for a free, unauthenticated integration: no platform (Netflix, Prime
    Video, Hotstar, Zee5, SonyLIV, ...) publishes a free API for linking
    straight into one specific title's page from outside their own app/site.
    """
    if movie_id in _live_providers_cache:
        return _live_providers_cache[movie_id]
    result = {"names": [], "link": None}
    if TMDB_API_KEY:
        try:
            resp = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
                params={"api_key": TMDB_API_KEY},
                timeout=8,
            )
            resp.raise_for_status()
            region_data = resp.json().get("results", {}).get("IN", {})
            names = []
            for category in ["flatrate", "free", "ads"]:
                for p in region_data.get(category, []):
                    names.append(p["provider_name"])
            result["names"] = sorted(set(names))
            result["link"] = region_data.get("link")
        except requests.exceptions.RequestException:
            pass
    _live_providers_cache[movie_id] = result
    return result


def retrieve_movies(query, user_age, travel_mode, excluded_ids=None, n=6, surprise=False,
                    location_state=None, recent_ids=None):
    excluded_ids = excluded_ids or set()
    recent_ids = {str(movie_id) for movie_id in (recent_ids or [])}
    direct_matches = find_title_matches(query)

    reference_movie = None
    if direct_matches and is_similarity_query(query):
        reference_movie = direct_matches[0]
        direct_matches = []

    if reference_movie and str(reference_movie["id"]) in id_to_row:
        query_vector = tfidf_matrix[id_to_row[str(reference_movie["id"])]]
    else:
        query_vector = vectorizer.transform([query])

    fetch_count = min((n + len(excluded_ids) + len(recent_ids) + 2) * 4, len(index_ids))
    sims = cosine_similarity(query_vector, tfidf_matrix)[0]
    general_positions = list(sims.argsort()[::-1][:fetch_count])

    target_genre = None if reference_movie else detect_target_genre(query)
    if target_genre:
        genre_positions = [
            pos for pos, mid in enumerate(index_ids)
            if movie_has_genre(movies_by_id.get(mid, {}), target_genre)
        ]
        # Rank the movies that actually carry this genre first, best
        # text-similarity first, instead of ranking the whole catalogue and
        # hoping the right genre floats to the top on its own. But a genre
        # like Horror is a small, often adult-certified slice of the
        # catalogue -- if we restricted the pool to ONLY that genre and it
        # got filtered down by age rating / already-watched / already-shown,
        # there could be nothing left at all ("No new matches"), even though
        # the catalogue clearly has other decent, age-appropriate picks. So
        # genre-correct movies come first, and the general ranking fills in
        # any remaining slots rather than leaving the user with nothing.
        genre_positions.sort(key=lambda pos: sims[pos], reverse=True)
        genre_positions = genre_positions[:fetch_count]
        seen_positions = set(genre_positions)
        top_positions = genre_positions + [
            pos for pos in general_positions if pos not in seen_positions
        ]
    else:
        top_positions = general_positions

    semantic_raw = [
        (index_ids[pos], movies_by_id.get(index_ids[pos], {}))
        for pos in top_positions
    ]
    # NOTE: top_positions is already sorted best-match-first (argsort on
    # cosine similarity). We used to unconditionally shuffle this list,
    # which threw away that relevance ordering and let a weakly-matching
    # movie from near the bottom of the candidate pool get picked over a
    # much better match — this was the main cause of odd/irrelevant picks
    # showing up (e.g. action or comedy titles under "Romance"). Only
    # shuffle for an actual "surprise me" request, where variety matters
    # more than strict ranking.
    if surprise:
        random.shuffle(semantic_raw)

    def age_ok(m):
        return get_min_age(m["certification"]) <= user_age

    seen_titles = set()
    blocked_count = 0

    direct_kept = []
    for m in direct_matches:
        norm = normalize_movie(m, movie_id=m["id"])
        if norm["id"] in excluded_ids or norm["id"] in recent_ids:
            continue
        if not age_ok(norm):
            blocked_count += 1
            continue
        if norm["title"] not in seen_titles:
            direct_kept.append(norm)
            seen_titles.add(norm["title"])

    semantic_kept = []
    ref_title = reference_movie["title"] if reference_movie else None
    for mid, meta in semantic_raw:
        norm = normalize_movie(meta, movie_id=mid)
        if norm["id"] in excluded_ids or norm["id"] in recent_ids:
            continue
        if ref_title and norm["title"] == ref_title:
            continue
        if not age_ok(norm):
            blocked_count += 1
            continue
        if norm["title"] not in seen_titles:
            semantic_kept.append(norm)
            seen_titles.add(norm["title"])

    if travel_mode:
        semantic_kept.sort(key=lambda m: 0 if is_light_genre(m) else 1)

    location_kept = []
    profile = get_location_profile(location_state) if is_location_query(query) else None
    if profile:
        location_titles = set()
        regional_candidates = []
        for movie in all_movies:
            score = location_match_score(movie, profile)
            if score:
                regional_candidates.append((score, movie))
        random.shuffle(regional_candidates)
        regional_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, movie in regional_candidates:
            norm = normalize_movie(movie, movie_id=movie["id"])
            if norm["id"] in excluded_ids or norm["id"] in recent_ids:
                continue
            if not age_ok(norm):
                blocked_count += 1
                continue
            if norm["title"] not in location_titles:
                location_kept.append(norm)
                location_titles.add(norm["title"])
            if len(location_kept) == 3:
                break

    combined = []
    combined_titles = set()
    for movie in location_kept + direct_kept + semantic_kept:
        if movie["title"] not in combined_titles:
            combined.append(movie)
            combined_titles.add(movie["title"])
        if len(combined) == n:
            break
    return combined, blocked_count


def ask_ai(user_query, movies, history, user_age, travel_mode, is_info=False):
    movie_list_text = "\n".join([
        f"- {m['title']} | Genres: {m['genres']} | Certification: {m['certification']} | "
        f"Available on: {m['watch_providers'] or 'not listed'} | Rating: {m['rating']} | "
        f"Year: {m.get('release_date', '')[:4] or 'unknown'} | "
        f"Language: {m.get('original_language_name') or m.get('original_language') or 'unknown'} | "
        f"Overview: {m.get('overview') or 'not available'}"
        for m in movies
    ])

    travel_note = (
        "\nThe user is currently travelling / on the go, so lean toward lighter, "
        "easy-to-follow, fun picks rather than long or heavy dramas, and briefly "
        "mention that these are good picks for travel."
        if travel_mode else ""
    )

    system_prompt = f"""You are MovieMate, a friendly movie recommendation assistant.
The current user is {user_age} years old, and every movie in the list below has
already been checked to be age-appropriate for them, so you don't need to filter
anything yourself.{travel_note}
Pay attention to earlier messages in this conversation."""

    if is_info:
        instruction = """The user is asking for full details about a movie already discussed above.
Write a richer, warm paragraph (4-6 sentences) covering what the movie is about, its tone and genre,
and why it's worth watching. Do NOT give just 1-2 short lines. Do NOT mention certification, streaming
platform, or rating yourself — that information is shown automatically below your response, so repeating
it would be redundant. Just focus on rich, specific plot and appeal details."""
    else:
        instruction = """If this is a general mood, genre, or "movies like X" request, reply with ONE
short, warm sentence reacting to what they asked for. Do NOT list movie titles, reasons, ratings, or
platforms yourself — that information is shown separately as movie cards below your message, so listing
it yourself would be redundant. If this is a specific follow-up question about one movie, answer that
question directly and briefly instead."""

    user_prompt = f"""The user just said: "{user_query}"

Relevant, age-appropriate movies retrieved from the database for this message:
{movie_list_text}

{instruction}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": user_prompt})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.7}

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/session")
def session_state():
    if "user_id" not in session:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "username": session.get("username"),
        "travel_mode": bool(session.get("travel_mode", False)),
        "location_state": session.get("location_state"),
        "has_started": bool(session.get("chat_started", False)),
    })


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    age = data.get("age")
    security_question = (data.get("security_question") or "").strip()
    security_answer = (data.get("security_answer") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400
    if not isinstance(age, int) or age < 1 or age > 120:
        return jsonify({"error": "Please enter a valid age."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password should be at least 6 characters."}), 400
    if not security_question or not security_answer:
        return jsonify({"error": "Please select a security question and provide an answer."}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    if cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"error": "That username is already taken."}), 400

    password_hash = generate_password_hash(password)
    answer_hash = generate_password_hash(security_answer.lower())
    cur.execute(
        """INSERT INTO users (username, password_hash, age, security_question, security_answer_hash, created_at)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (username, password_hash, age, security_question, answer_hash, datetime.utcnow()),
    )
    user_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()

    session["user_id"] = user_id
    session["username"] = username
    session["user_age"] = age
    session["travel_mode"] = False
    session["location_state"] = None
    session["chat_started"] = False
    session["last_movies"] = []
    session["recommendation_history"] = {}

    return jsonify({"message": "ok", "username": username, "age": age})


@app.route("/login", methods=["POST"])
def login():
    if is_rate_limited(f"login:{request.remote_addr}"):
        return jsonify({"error": "Too many attempts. Please wait a minute and try again."}), 429

    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Incorrect username or password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["user_age"] = user["age"]
    session["travel_mode"] = False
    session["location_state"] = None
    session["chat_started"] = False
    session["last_movies"] = []
    session["recommendation_history"] = {}

    return jsonify({"message": "ok", "username": user["username"], "age": user["age"]})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "ok"})


@app.route("/forgot-password/question", methods=["POST"])
def forgot_password_question():
    data = request.get_json()
    username = (data.get("username") or "").strip()

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT security_question FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user or not user["security_question"]:
        return jsonify({"error": "No account found with that username."}), 404

    return jsonify({"question": user["security_question"]})


@app.route("/forgot-password/reset", methods=["POST"])
def forgot_password_reset():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    answer = (data.get("answer") or "").strip().lower()
    new_password = data.get("new_password") or ""

    if len(new_password) < 6:
        return jsonify({"error": "New password should be at least 6 characters."}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()

    if not user or not user["security_answer_hash"] or not check_password_hash(user["security_answer_hash"], answer):
        cur.close()
        conn.close()
        return jsonify({"error": "Incorrect answer."}), 401

    new_hash = generate_password_hash(new_password)
    cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, user["id"]))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "ok"})


@app.route("/start", methods=["POST"])
def start():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    data = request.get_json() or {}
    travel_mode = bool(data.get("travel_mode"))
    session["location_state"] = (data.get("location_state") or "").strip() or None
    session["travel_mode"] = travel_mode
    session["chat_started"] = True
    session["last_movies"] = []
    return jsonify({"message": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "Session expired, please log in again."}), 401

    if is_rate_limited(f"chat:{session['user_id']}"):
        return jsonify({"error": "You're sending messages too fast. Please slow down a little."}), 429

    data = request.get_json()
    user_query = (data.get("message") or "").strip()
    if not user_query:
        return jsonify({"error": "Please type something."}), 400

    user_id = session["user_id"]
    user_age = session["user_age"]
    travel_mode = session.get("travel_mode", False)
    location_state = session.get("location_state")
    history = get_recent_history(user_id)
    excluded_ids = get_excluded_ids(user_id)
    last_movies = session.get("last_movies", [])

    info_query = is_info_query(user_query)

    is_movie_followup = info_query and bool(last_movies)
    if is_movie_followup:
        num = extract_number(user_query)
        if num and 1 <= num <= len(last_movies):
            movies = [last_movies[num - 1]]
        else:
            movies = last_movies
        blocked_count = 0
    else:
        query_key = user_query.lower()
        recommendation_history = session.get("recommendation_history", {})
        recent_ids = recommendation_history.get(query_key, [])
        movies, blocked_count = retrieve_movies(
            user_query,
            user_age,
            travel_mode,
            excluded_ids,
            n=6,
            location_state=location_state,
            recent_ids=recent_ids,
        )
        recommendation_history.pop(query_key, None)
        recommendation_history[query_key] = (
            recent_ids + [movie["id"] for movie in movies]
        )[-18:]
        while len(recommendation_history) > 6:
            recommendation_history.pop(next(iter(recommendation_history)))
        session["recommendation_history"] = recommendation_history

    session["last_movies"] = movies
    response_token = uuid.uuid4().hex
    _pending_chat_requests[response_token] = {
        "user_id": user_id,
        "user_query": user_query,
        "movies": movies,
        "history": history,
        "user_age": user_age,
        "travel_mode": travel_mode,
        "is_info": info_query,
    }

    return jsonify({"movies": movies, "blocked_count": blocked_count, "response_token": response_token})


@app.route("/chat/response", methods=["POST"])
def chat_response():
    if "user_id" not in session:
        return jsonify({"error": "Session expired, please log in again."}), 401

    data = request.get_json() or {}
    response_token = data.get("response_token")
    pending = _pending_chat_requests.pop(response_token, None)
    if not pending or pending["user_id"] != session["user_id"]:
        return jsonify({"error": "This response expired. Please send your message again."}), 400

    try:
        answer = ask_ai(
            pending["user_query"], pending["movies"], pending["history"],
            pending["user_age"], pending["travel_mode"], is_info=pending["is_info"]
        )
    except requests.exceptions.HTTPError as e:
        response = e.response
        detail = "The AI provider rejected the request. Check GROQ_API_KEY and GROQ_MODEL."
        if response is not None:
            try:
                provider_error = response.json().get("error", {}).get("message")
                if provider_error:
                    detail = f"AI service error: {provider_error}"
            except (ValueError, TypeError):
                pass
        return jsonify({"error": detail}), 502
    except requests.exceptions.RequestException:
        return jsonify({"error": "The AI service could not be reached. Check the deployment network and Groq status."}), 502
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "The AI service returned an unexpected response."}), 502

    save_message(session["user_id"], "user", pending["user_query"])
    save_message(session["user_id"], "assistant", answer)
    return jsonify({"reply": answer})


@app.route("/trailers", methods=["POST"])
def trailers():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    data = request.get_json() or {}
    movie_ids = [str(movie_id) for movie_id in data.get("movie_ids", [])[:10] if movie_id]
    if not movie_ids:
        return jsonify({"trailers": {}, "providers": {}})

    # NOTE: this used to submit two OUTER jobs to the pool, each of which then
    # called executor.map(...) on that very same pool and blocked waiting for
    # its results. That's fine as long as there are still free worker threads
    # left over to run the inner jobs -- but for a single movie (exactly what
    # happens when a movie is added to the watchlist one at a time), the pool
    # was sized to 2 workers, both outer jobs immediately claimed both of
    # them, and the inner jobs they were each waiting on could never get a
    # worker to run on. That's a permanent deadlock: the request just hangs
    # until the server's own request timeout kills it, the frontend gets a
    # non-JSON timeout page back, and the Trailer/Watch Now buttons show
    # "Unavailable". The main recommendations grid never hit this because it
    # always requests 6 movies at once, which happened to size the pool large
    # enough to avoid the deadlock.
    #
    # Fix: submit every trailer/provider lookup directly and flatly to one
    # pool, with no job ever waiting on another job in the same pool.
    with ThreadPoolExecutor(max_workers=min(20, len(movie_ids) * 2)) as executor:
        trailer_futures = {mid: executor.submit(get_trailer_url, mid) for mid in movie_ids}
        provider_futures = {mid: executor.submit(get_live_providers, mid) for mid in movie_ids}
        trailer_urls = {mid: future.result() for mid, future in trailer_futures.items()}
        live_providers = {mid: future.result() for mid, future in provider_futures.items()}
    return jsonify({
        "trailers": trailer_urls,
        "providers": live_providers,
    })


@app.route("/exclude", methods=["POST"])
def exclude():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    data = request.get_json()
    movie_id = str(data.get("movie_id", ""))
    movie_title = data.get("movie_title", "")
    if not movie_id:
        return jsonify({"error": "Missing movie_id."}), 400
    add_watched(session["user_id"], movie_id, movie_title)
    return jsonify({"message": "excluded"})


@app.route("/watchlist", methods=["GET"])
def watchlist():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    try:
        return jsonify({"movies": get_watchlist(session["user_id"])})
    except psycopg2.Error:
        app.logger.exception("Could not load watchlist")
        return jsonify({"error": "The watchlist database is unavailable. Check DATABASE_URL and the Render logs."}), 503


@app.route("/watchlist", methods=["POST"])
def add_watchlist_movie():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    movie = request.get_json() or {}
    if not movie.get("id") or not movie.get("title"):
        return jsonify({"error": "Movie details are incomplete."}), 400
    try:
        add_to_watchlist(session["user_id"], movie)
    except psycopg2.Error:
        app.logger.exception("Could not add movie to watchlist")
        return jsonify({"error": "The watchlist database is unavailable. Check DATABASE_URL and the Render logs."}), 503
    return jsonify({"message": "added", "movie": movie})


@app.route("/watchlist/<movie_id>", methods=["DELETE"])
def remove_watchlist_movie(movie_id):
    if "user_id" not in session:
        return jsonify({"error": "Please log in first."}), 401
    try:
        remove_from_watchlist(session["user_id"], movie_id)
    except psycopg2.Error:
        app.logger.exception("Could not remove movie from watchlist")
        return jsonify({"error": "The watchlist database is unavailable. Check DATABASE_URL and the Render logs."}), 503
    return jsonify({"message": "removed"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
