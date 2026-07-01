"""
chunking/chunker.py
Splits large documents into chunks.
Hadith + Quran ayahs are already natural chunks,
so this mainly handles future PDFs/books.
Also adds chunk_index metadata for citation.
"""

import json
import os

os.makedirs("datasets/chunks", exist_ok=True)


def chunk_document(doc: dict, max_chars: int = 1000, overlap: int = 100) -> list:
    """
    If doc text is under max_chars → single chunk, no split.
    If over → split with overlap so context isn't lost at boundaries.
    """
    text = doc["text"]

    if len(text) <= max_chars:
        chunk = doc.copy()
        chunk["chunk_index"] = 0
        chunk["chunk_total"] = 1
        return [chunk]

    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end  = start + max_chars
        part = text[start:end]

        chunk = doc.copy()
        chunk["text"]        = part
        chunk["chunk_index"] = idx
        chunk["id"]          = f"{doc['id']}_chunk{idx}"

        chunks.append(chunk)
        start += max_chars - overlap
        idx   += 1

    for c in chunks:
        c["chunk_total"] = idx

    return chunks


def chunk_dataset(input_path: str, output_path: str, label: str):
    print(f"Chunking {label}...")

    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"  {len(docs):,} docs → {len(all_chunks):,} chunks → {output_path}")
    return all_chunks


if __name__ == "__main__":
    hadith_chunks = chunk_dataset(
        "datasets/processed/hadith_docs.json",
        "datasets/chunks/hadith_chunks.json",
        "Hadith"
    )
    quran_chunks = chunk_dataset(
        "datasets/processed/quran_docs.json",
        "datasets/chunks/quran_chunks.json",
        "Quran"
    )

    print(f"\nTotal chunks ready for embedding: "
          f"{len(hadith_chunks) + len(quran_chunks):,}")
    print("Next step: python embeddings/embedder.py")