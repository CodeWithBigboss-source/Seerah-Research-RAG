"""
preprocessing/cleaner.py
Loads raw Hadith + Quran datasets, cleans text,
outputs standardized document objects for chunking.
"""

import json
import re
import os
import unicodedata

os.makedirs("datasets/processed", exist_ok=True)


# ── Text cleaning ─────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Normalize unicode (handles Arabic diacritics consistently)
    text = unicodedata.normalize("NFKC", text)
    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_id(source: str, index: int) -> str:
    return f"{source}_{index:06d}"


# ══════════════════════════════════════════════════════════
# HADITH PROCESSOR
# Input:  datasets/raw/hadith/hadith_all.json
# Output: datasets/processed/hadith_docs.json
#
# Each doc = one hadith (already a natural chunk)
# Key field: grading → powers Truth Verification Mode
# ══════════════════════════════════════════════════════════

def process_hadith():
    print("Processing Hadith dataset...")
    path = "datasets/raw/hadith/hadith_all.json"

    if not os.path.exists(path):
        print(f"  ✗ Not found: {path}")
        return []

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Clean both language fields
            en_text  = clean_text(entry.get("English_Text", "")  or
                                   entry.get("english", "")       or
                                   entry.get("text", "")          or "")
            ar_text  = clean_text(entry.get("Arabic_Text", "")   or
                                   entry.get("arabic", "")        or "")
            book     = clean_text(entry.get("Book", "")          or
                                   entry.get("book", "")          or "Unknown")
            grading  = clean_text(entry.get("Grade", "")         or
                                   entry.get("grade", "")         or
                                   entry.get("Grading", "")       or "Unknown")
            chapter  = clean_text(entry.get("Chapter_Title_English", "") or "")
            ref_no   = str(entry.get("Hadith_number", "") or
                           entry.get("hadith_number", "") or
                           entry.get("id", i))

            if not en_text:
                continue

            docs.append({
                "id":          make_id("hadith", i),
                "source_type": "hadith",
                "text":        en_text,          # used for embedding
                "arabic":      ar_text,
                "metadata": {
                    "book":      book,
                    "chapter":   chapter,
                    "reference": f"{book} {ref_no}",
                    "grading":   grading,
                    "language":  "en",
                }
            })

    out = "datasets/processed/hadith_docs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {len(docs):,} hadith documents saved → {out}")
    print(f"  Sample:")
    print(f"    text:      {docs[0]['text'][:120]}...")
    print(f"    book:      {docs[0]['metadata']['book']}")
    print(f"    grading:   {docs[0]['metadata']['grading']}")
    return docs


# ══════════════════════════════════════════════════════════
# QURAN + TAFSEER PROCESSOR
# Input:  datasets/raw/quran/quran_tafseer.json
# Output: datasets/processed/quran_docs.json
#
# Each doc = one ayah (verse) — natural chunk boundary
# Tafseer included as extended context
# ══════════════════════════════════════════════════════════

def process_quran():
    print("\nProcessing Quran + Tafseer dataset...")
    path = "datasets/raw/quran/quran_tafseer.json"

    if not os.path.exists(path):
        print(f"  ✗ Not found: {path}")
        return []

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try common column name variants
            surah    = str(entry.get("surah_number", "")   or
                           entry.get("chapter", "")         or
                           entry.get("sura", "")            or "")
            ayah     = str(entry.get("ayah_number", "")    or
                           entry.get("verse", "")           or
                           entry.get("aya", "")             or "")
            ar_text  = clean_text(entry.get("arabic_text", "")  or
                                   entry.get("arabic", "")       or "")
            en_text  = clean_text(entry.get("english_text", "") or
                                   entry.get("translation", "")  or
                                   entry.get("text", "")         or "")
            tafseer  = clean_text(entry.get("tafseer", "")      or
                                   entry.get("tafsir", "")       or