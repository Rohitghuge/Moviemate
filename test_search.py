"""
test_search.py
A quick way to test that your ChromaDB movie index actually works.
Type in a mood or description, and it will show the 5 closest-matching
movies based on meaning (not just keyword matching).

This does NOT use any AI chatbot yet — it's just testing the retrieval
(the "R" in RAG) before we add the AI response generation on top.
"""

from sentence_transformers import SentenceTransformer
import chromadb

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to your movie index...")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="movies")

print(f"Index loaded with {collection.count()} movies.\n")

while True:
    query = input("What kind of movie are you in the mood for? (or type 'quit' to exit)\n> ")
    if query.strip().lower() == "quit":
        break

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)

    print("\nTop 5 matches:\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        print(f"{i + 1}. {meta['title']}  ({meta['genres']})")
        print(f"   Certification: {meta['certification']}  |  Rating: {meta['rating']}")
        print(f"   Available on: {meta['watch_providers'] or 'Not listed for India'}")
        print()

    print("-" * 50 + "\n")
