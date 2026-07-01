"""
embeddings/embedder.py
Embeds all chunks and stores them in ChromaDB.
Model: BAAI/bge-m3 — multilingual (English + Urdu + Arabic)
Runs on T4 GPU automatically.
"""

import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"        # multilingual, best for this task
BATCH_SIZE   = 64                    # fits T4 16GB comfortably
CHROMA_PATH  = "vector_db/chroma"
COLLECTION   = "islamic_knowledge"

os.makedirs(CHROMA_PATH, exist_ok=True)


def load_chunks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_store(chunks: list, model, collection, label: str):
    print(f"\nEmbedding {label} — {len(chunks):,} chunks...")

    texts     = [c["text"]     for c in chunks]
    ids       = [c["id"]       for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Add source_type to metadata for filtering later
    for i, chunk in enumerate(chunks):
        metadatas[i]["source_type"]  = chunk.get("source_type", "unknown")
        metadatas[i]["chunk_index"]  = str(chunk.get("chunk_index", 0))
        metadatas[i]["chunk_total"]  = str(chunk.get("chunk_total", 1))

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  Embedding {label}"):
        batch = texts[i : i + BATCH_SIZE]
        embs  = model.encode(
            batch,
            normalize_embeddings=True,   # cosine similarity ready
            show_progress_bar=False
        )
        all_embeddings.extend(embs.tolist())

    # Store in ChromaDB in batches (ChromaDB has a 41k limit per add)
    CHROMA_BATCH = 5000
    for i in tqdm(range(0, len(chunks), CHROMA_BATCH), desc=f"  Storing {label}"):
        collection.add(
            ids        = ids[i : i + CHROMA_BATCH],
            embeddings = all_embeddings[i : i + CHROMA_BATCH],
            documents  = texts[i : i + CHROMA_BATCH],
            metadatas  = metadatas[i : i + CHROMA_BATCH],
        )

    print(f"  ✓ {len(chunks):,} chunks stored in ChromaDB")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading embedding model: {EMBED_MODEL}")

    model  = SentenceTransformer(EMBED_MODEL, device=device)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    hadith_chunks   = load_chunks("datasets/chunks/hadith_chunks.json")
    quran_ar_chunks = load_chunks("datasets/chunks/quran_ar_chunks.json")
    quran_en_chunks = load_chunks("datasets/chunks/quran_en_chunks.json")

    embed_and_store(hadith_chunks,   model, collection, "Hadith")
    embed_and_store(quran_ar_chunks, model, collection, "Quran Arabic")
    embed_and_store(quran_en_chunks, model, collection, "Quran English")

    print(f"\n{'='*50}")
    print(f"EMBEDDING COMPLETE")
    print(f"  Total vectors in ChromaDB: {collection.count():,}")
    print(f"{'='*50}")
    print("Next: python retriever/retriever.py")