"""
embeddings/embedder.py
Streaming embedder with checkpoint recovery.
Reads chunks in batches, embeds, stores immediately, frees GPU memory.
Never accumulates all embeddings in RAM.
"""

import json
import os
import gc
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE  = 512                    # chunks per embedding batch
CHROMA_PATH = os.path.join(
    os.environ.get("SEERAH_STORAGE", "datasets"),
    "vector_db", "chroma"
)
COLLECTION  = "islamic_knowledge"

CHUNK_FILES = [
    os.path.join(os.environ.get("SEERAH_STORAGE", "datasets"), "chunks", "hadith_chunks.json"),
    os.path.join(os.environ.get("SEERAH_STORAGE", "datasets"), "chunks", "quran_ar_chunks.json"),
    os.path.join(os.environ.get("SEERAH_STORAGE", "datasets"), "chunks", "quran_en_chunks.json"),
]


# ── Generator — yields one batch at a time ────────────────

def batch_generator(chunk_files: list, batch_size: int, existing_ids: set):
    """
    Reads chunk files one line at a time.
    Yields batches of chunks not already in ChromaDB.
    Never loads entire file into memory.
    """
    batch = []
    skipped = 0
    total_yielded = 0

    for file_path in chunk_files:
        if not os.path.exists(file_path):
            print(f"  Skipping missing file: {file_path}")
            continue

        print(f"\n  Reading: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            chunk_id = chunk["id"]

            # Skip already embedded chunks (checkpoint recovery)
            if chunk_id in existing_ids:
                skipped += 1
                continue

            batch.append(chunk)

            if len(batch) >= batch_size:
                total_yielded += len(batch)
                yield batch
                batch = []

        # Yield remaining chunks from this file
        if batch:
            total_yielded += len(batch)
            yield batch
            batch = []

    if skipped:
        print(f"\n  Checkpoint: skipped {skipped:,} already-embedded chunks")
    print(f"  Total new chunks to embed: {total_yielded:,}")


# ── Embed one batch and store immediately ─────────────────

def embed_batch(batch: list, model, collection):
    texts     = [c["text"]                    for c in batch]
    ids       = [c["id"]                      for c in batch]
    metadatas = []

    for c in batch:
        meta = c.get("metadata", {}).copy()
        meta["source_type"] = c.get("source_type", "unknown")
        meta["chunk_index"] = str(c.get("chunk_index", 0))
        meta["chunk_total"] = str(c.get("chunk_total", 1))
        # ChromaDB requires all values to be strings
        meta = {k: str(v) for k, v in meta.items()}
        metadatas.append(meta)

    # Embed
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    ).tolist()

    # Store immediately
    CHROMA_BATCH = 5000
    for i in range(0, len(batch), CHROMA_BATCH):
        collection.add(
            ids        = ids[i:i+CHROMA_BATCH],
            embeddings = embeddings[i:i+CHROMA_BATCH],
            documents  = texts[i:i+CHROMA_BATCH],
            metadatas  = metadatas[i:i+CHROMA_BATCH],
        )

    # Free GPU memory immediately
    del embeddings, texts
    gc.collect()
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:       {device}")
    print(f"Chroma path:  {CHROMA_PATH}")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print("Model loaded.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    print(f"Existing vectors in ChromaDB: {existing_count:,}")

    # Get existing IDs for checkpoint recovery
    existing_ids = set()
    if existing_count > 0:
        print("Loading existing IDs for checkpoint recovery...")
        result = collection.get(include=[])
        existing_ids = set(result["ids"])
        print(f"  Loaded {len(existing_ids):,} existing IDs")

    # Stream and embed
    batch_num = 0
    for batch in tqdm(
        batch_generator(CHUNK_FILES, BATCH_SIZE, existing_ids),
        desc="Embedding batches"
    ):
        embed_batch(batch, model, collection)
        batch_num += 1

        # Print progress every 10 batches
        if batch_num % 10 == 0:
            print(f"  Batches done: {batch_num} | "
                  f"Vectors stored: {collection.count():,}")

    final_count = collection.count()
    print(f"\n{'='*50}")
    print(f"EMBEDDING COMPLETE")
    print(f"  Total vectors in ChromaDB: {final_count:,}")
    print(f"  New vectors added:         {final_count - existing_count:,}")
    print(f"  ChromaDB path:             {CHROMA_PATH}")
    print(f"{'='*50}")
    print("Next: python retriever/retriever.py")