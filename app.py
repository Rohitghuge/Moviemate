"""
MovieMate - Heroku version with PostgreSQL.
All comments explain what each part does.
"""

import os
import json
import re
import difflib
import psycopg2                     # For PostgreSQL (Heroku)
import psycopg2.extras               # For dictionary cursors
import requests
from flask import Flask, request, jsonify, session, render_template
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
import chromadb

# ---------- Load environment variables ----------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
DATABASE_URL = os.getenv("DATABASE_URL")   # Heroku provides this

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
HISTORY_LIMIT = 6

LIGHT_GENRES = {"Comedy", "Animation", "Adventure", "Family", "Music", "Fantasy"}
SIMILARITY_TRIGGERS = ["like ", "similar to", "similar", "such as", "in the style of", "in the vein of"]

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

# ---------- PostgreSQL Database Helpers ----------
def get_db():
    """Return a connection to the PostgreSQL database."""
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create the users and messages tables if they don't exist."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER NOT NULL,
            travel_mode BOOLEAN DEFAULT FALSE,
            location TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Run this once when the app starts
init_db()

def create_user(username, password, age, travel_mode):
    """Hash password and save new user."""
    hashed = generate_password_hash(password)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password, age, travel_mode) VALUES (%s, %s, %s, %s)",
        (username, hashed, age, travel_mode)
    )
    conn.commit()
    conn.close()

def get_user_by_username(username):
    """Fetch a user by username."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    """Fetch a user by ID."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user

def save_message(user_id, role, content):
    """Save a chat message (user or assistant) to the database."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (%s, %s, %s)",
        (user_id, role, content)
    )
    conn.commit()
    conn.close()

def get_user_messages(user_id, limit=50):
    """Get the last `limit` messages for a user, oldest first."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT role, content FROM messages WHERE user_id = %s ORDER BY timestamp ASC LIMIT %s",
        (user_id, limit)
    )
    msgs = cur.fetchall()
    conn.close()
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

# ---------- Load movie data and embedding index ----------
print("Loading movie list...")
with open("movies.json", "r", encoding="utf-8") as f:
    all_movies = json.load(f)
all_titles_lower = [m["title"].lower() for m in all_movies]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to movie index...")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="movies")
print(f"Ready! {collection.count()} movies loaded.")

# ---------- Flask app ----------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ---------- Movie helper functions (unchanged) ----------
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

def normalize_movie(m, movie_id=None):
    genres = m.get("genres", "")
    genres_str = ", ".join(genres) if isinstance(genres, list) else (genres or "")
    providers = m.get("watch_providers", "")
    providers_str = ", ".join(providers) if isinstance(providers, list) else (providers or "")
    return {
        "id": str(movie_id if movie_id is not None else m.get("id", "")),
        "title": m.get("title") or "Unknown",
        "genres": genres_str,
        "certification": m.get("certification") or "NR",
        "watch_providers": providers_str,
        "release_date": m.get("release_date") or "",
        "rating": m.get("rating") or 0,
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

INFO_TRIGGERS = ["info", "detail", "tell me more", "more about", "elaborate", "know more"]
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

def retrieve_movies(query, user_age, travel_mode, excluded_ids=None, n=5):
    excluded_ids = excluded_ids or set()
    direct_matches = find_title_matches(query)

    reference_movie = None
    if direct_matches and is_similarity_query(query):
        reference_movie = direct_matches[0]
        direct_matches = []

    if reference_movie:
        stored = collection.get(ids=[str(reference_movie["id"])], include=["embeddings"])
        if stored and len(stored.get("embeddings", [])) > 0:
            query_embedding = [stored["embeddings"][0]]
        else:
            query_embedding = model.encode([query]).tolist()
    else:
        query_embedding = model.encode([query]).tolist()

    fetch_count = (n + len(excluded_ids) + 2) * 4
    results = collection.query(query_embeddings=query_embedding, n_results=fetch_count)
    semantic_raw = list(zip(results["ids"][0], results["metadatas"][0]))

    def age_ok(m):
        return get_min_age(m["certification"]) <= user_age

    seen_titles = set()
    blocked_count = 0

    direct_kept = []
    for m in direct_matches:
        norm = normalize_movie(m, movie_id=m["id"])
        if norm["id"] in excluded_ids:
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
        if norm["id"] in excluded_ids:
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

    combined = (direct_kept + semantic_kept)[:n]
    return combined, blocked_count

def ask_ai(user_query, movies, history, user_age, travel_mode, is_info=False):
    movie_list_text = "\n".join([
        f"- {m['title']} | Genres: {m['genres']} | Certification: {m['certification']} | "
        f"Available on: {m['watch_providers'] or 'not listed'} | Rating: {m['rating']}"
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
        instruction = """If this is a general mood, genre, or "movies like X" request, reply with one
short warm sentence, then a numbered list of up to 5 movies from the list above, each with a one-line
reason and where to watch it. If this is a specific follow-up question about one movie, just answer that
question directly and briefly, don't repeat a full list. Only reference movies from the list above or
from earlier in this conversation."""

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

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    age = data.get("age")
    travel_mode = bool(data.get("travel_mode", False))

    if not username or not password:
        return jsonify({"error": "Username and password required."}), 400
    if not isinstance(age, int) or age < 1 or age > 120:
        return jsonify({"error": "Please enter a valid age."}), 400

    if get_user_by_username(username):
        return jsonify({"error": "Username already taken."}), 400

    create_user(username, password, age, travel_mode)
    return jsonify({"message": "Account created! Please log in."})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = get_user_by_username(username)
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid username or password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"message": "Login successful", "username": user["username"]})

@app.route("/set_travel_and_location", methods=["POST"])
def set_travel_and_location():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in."}), 401

    data = request.get_json() or {}
    travel_mode = bool(data.get("travel_mode", False))
    location = (data.get("location") or "").strip()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET travel_mode = %s, location = %s WHERE id = %s",
        (travel_mode, location, user_id)
    )
    conn.commit()
    conn.close()

    session["travel_mode"] = travel_mode
    return jsonify({"message": "Updated", "travel_mode": travel_mode, "location": location})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out."})

@app.route("/whoami", methods=["GET"])
def whoami():
    username = session.get("username")
    if username:
        return jsonify({"username": username})
    return jsonify({"error": "Not logged in"}), 401

@app.route("/load_chat", methods=["GET"])
def load_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in."}), 401
    messages = get_user_messages(user_id, limit=50)
    return jsonify({"messages": messages})

@app.route("/chat", methods=["POST"])
def chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Please log in first."}), 401

    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"error": "User not found."}), 401

    user_age = user["age"]
    travel_mode = bool(user["travel_mode"])

    data = request.get_json()
    user_query = (data.get("message") or "").strip()
    if not user_query:
        return jsonify({"error": "Please type something."}), 400

    excluded_ids = set(session.get("excluded_ids", []))
    info_query = is_info_query(user_query)
    history = get_user_messages(user_id, limit=HISTORY_LIMIT)
    history_for_ai = [{"role": m["role"], "content": m["content"]} for m in history]

    movies, blocked_count = retrieve_movies(user_query, user_age, travel_mode, excluded_ids)

    for m in movies:
        m["trailer_url"] = get_trailer_url(m["id"])

    try:
        answer = ask_ai(user_query, movies, history_for_ai, user_age, travel_mode, is_info=info_query)
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"AI service error: {e}"}), 500

    save_message(user_id, "user", user_query)
    save_message(user_id, "assistant", answer)

    session["last_movies"] = movies

    return jsonify({"reply": answer, "movies": movies, "blocked_count": blocked_count})

@app.route("/exclude", methods=["POST"])
def exclude():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in."}), 401

    data = request.get_json()
    movie_id = str(data.get("movie_id", ""))
    if not movie_id:
        return jsonify({"error": "Missing movie_id."}), 400

    excluded_ids = session.get("excluded_ids", [])
    if movie_id not in excluded_ids:
        excluded_ids.append(movie_id)
    session["excluded_ids"] = excluded_ids

    return jsonify({"message": "excluded", "excluded_ids": excluded_ids})

if __name__ == "__main__":
    app.run(debug=True, port=5000)