"""
chatbot.py
The core RAG chatbot: asks the user's age and whether they're travelling once
at the start, then takes what they type, retrieves the closest-matching
movies (direct/fuzzy title match + meaning), filters out anything above their
age certification, and — if travelling — prefers lighter, easier-to-watch
genres. Sends the result plus recent conversation history to a free AI model
(via Groq) which writes a natural, conversational recommendation.
"""

import os
import json
import difflib
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
HISTORY_LIMIT = 6  # keep last 3 user+assistant exchanges

LIGHT_GENRES = {"Comedy", "Animation", "Adventure", "Family", "Music", "Fantasy"}

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Check your .env file.")

print("Loading movie list...")
with open("movies.json", "r", encoding="utf-8") as f:
    all_movies = json.load(f)
all_titles_lower = [m["title"].lower() for m in all_movies]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to your movie index...")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="movies")
print(f"Ready! {collection.count()} movies loaded.\n")


def get_min_age(certification):
    """Map a certification string (Indian or US) to a minimum viewing age."""
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
    """Check if a movie belongs to an easy-watch genre, good for travel."""
    genres = movie.get("genres", "")
    genre_list = [g.strip() for g in genres.split(",")] if isinstance(genres, str) else genres
    return any(g in LIGHT_GENRES for g in genre_list)


def find_title_matches(query, limit=2):
    """Catch cases where the user typed (or nearly typed) a movie title directly."""
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


def retrieve_movies(query, user_age, travel_mode, n=10):
    """
    Direct/fuzzy title matches always come first (respecting explicit intent).
    Semantic matches fill the rest, reordered toward light genres if travelling.
    Anything above the user's age certification is filtered out entirely.
    """
    direct_matches = find_title_matches(query)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n * 4)
    semantic_matches = [results["metadatas"][0][i] for i in range(len(results["ids"][0]))]

    def age_ok(m):
        return get_min_age(m["certification"]) <= user_age

    seen_titles = set()
    blocked_count = 0

    direct_kept = []
    for m in direct_matches:
        if not age_ok(m):
            blocked_count += 1
            continue
        if m["title"] not in seen_titles:
            direct_kept.append(m)
            seen_titles.add(m["title"])

    semantic_kept = []
    for m in semantic_matches:
        if not age_ok(m):
            blocked_count += 1
            continue
        if m["title"] not in seen_titles:
            semantic_kept.append(m)
            seen_titles.add(m["title"])

    if travel_mode:
        semantic_kept.sort(key=lambda m: 0 if is_light_genre(m) else 1)

    combined = (direct_kept + semantic_kept)[:n]
    return combined, blocked_count


def ask_ai(user_query, movies, history, user_age, travel_mode):
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
Pay attention to earlier messages in this conversation. If the user asks a
follow-up question about a specific movie already mentioned (like "what about X"),
answer specifically and briefly about that movie instead of giving a whole new
unrelated list."""

    user_prompt = f"""The user just said: "{user_query}"

Relevant, age-appropriate movies retrieved from the database for this message:
{movie_list_text}

If this is a general mood or genre request, reply with one short warm sentence,
then a numbered list of up to 10 movies from the list above, each with a one-line
reason and where to watch it. If this is a specific follow-up question about one
movie, just answer that question directly and briefly, don't repeat a full list.
Only reference movies from the list above or from earlier in this conversation."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": user_prompt})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Welcome to MovieMate!\n")

    while True:
        age_input = input("What's your age? (used to filter age-appropriate movies)\n> ").strip()
        if age_input.isdigit():
            user_age = int(age_input)
            break
        print("Please enter a number, like 16 or 21.")

    travel_input = input("\nAre you travelling / on the go right now? (yes/no)\n> ").strip().lower()
    travel_mode = travel_input in ("yes", "y")

    print(f"\nGot it — filtering for age {user_age}+", end="")
    print(", and prioritizing light, easy-watch picks since you're travelling.\n" if travel_mode else ".\n")

    conversation_history = []

    while True:
        user_query = input("What are you in the mood to watch? (or type 'quit')\n> ")
        if user_query.strip().lower() == "quit":
            break

        print("\nSearching movies...")
        movies, blocked_count = retrieve_movies(user_query, user_age, travel_mode)

        if blocked_count:
            print(f"(Filtered out {blocked_count} result(s) not suitable for age {user_age})")

        print("Asking AI to write your recommendation...\n")
        try:
            answer = ask_ai(user_query, movies, conversation_history, user_age, travel_mode)
        except requests.exceptions.HTTPError as e:
            print(f"Groq API error: {e}")
            print("Check that your GROQ_API_KEY in .env is correct and still active.")
            continue

        print("MovieMate says:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")

        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": answer})
        conversation_history = conversation_history[-HISTORY_LIMIT:]
