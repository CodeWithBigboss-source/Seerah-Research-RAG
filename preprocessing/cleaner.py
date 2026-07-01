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


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_field(entry, *keys):
    """Try multiple key names, return first non-empty value."""
    for key in keys:
        val = entry.get(key, "")
        if val:
            return str(val)
    return ""


def make_id(source, index):
    return f"{source}_{index:06d}"


def process_hadith():
    print("Processing Hadith dataset...")
    path = "datasets/raw/hadith/hadith_all.json"

    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return []

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            en_text = clean_text(get_field(entry, "English_Text", "english", "text"))
            ar_text = clean_text(get_field(entry, "Arabic_Text", "arabic"))
            book    = clean_text(get_field(entry, "Book", "book")) or "Unknown"
            grading = clean_text(get_field(entry, "Grade", "grade", "Grading")) or "Unknown"
            chapter = clean_text(get_field(entry, "Chapter_Title_English", "chapter"))
            ref_no  = get_field(entry, "Hadith_number", "hadith_number", "id") or str(i)

            if not en_text:
                continue

            docs.append({
                "id":          make_id("hadith", i),
                "source_type": "hadith",
                "text":        en_text,
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

    print(f"  {len(docs):,} hadith documents saved to {out}")
    print(f"  Sample book:    {docs[0]['metadata']['book']}")
    print(f"  Sample grading: {docs[0]['metadata']['grading']}")
    print(f"  Sample text:    {docs[0]['text'][:100]}...")
    return docs


def process_quran():
    print("\nProcessing Quran Arabic Tafseer dataset...")
    path = "datasets/raw/quran/quran_tafseer.json"

    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return []

    docs = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Exact column names confirmed from inspection
            surah_name      = clean_text(entry.get("surah_name", ""))
            revelation_type = entry.get("revelation_type", "")
            ayah_arabic     = clean_text(entry.get("ayah", ""))
            tafsir_book     = clean_text(entry.get("tafsir_book", ""))
            tafsir_content  = clean_text(entry.get("tafsir_content", ""))

            # Build embedding text
            embedding_text = ayah_arabic
            if tafsir_content:
                embedding_text = ayah_arabic + " " + tafsir_content

            if not embedding_text.strip():
                skipped += 1
                continue

            docs.append({
                "id":          make_id("quran_ar", i),
                "source_type": "quran",
                "text":        embedding_text,
                "arabic":      ayah_arabic,
                "translation": "",
                "tafseer":     tafsir_content,
                "metadata": {
                    "surah":            surah_name,
                    "ayah":             str(i),
                    "surah_name":       surah_name,
                    "reference":        surah_name,
                    "tafsir_book":      tafsir_book,
                    "revelation_type":  revelation_type,
                    "grading":          "Authentic",
                    "language":         "ar",
                }
            })

    out = "datasets/processed/quran_docs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  {len(docs):,} Arabic Quran docs saved to {out}")
    print(f"  Skipped: {skipped}")
    if docs:
        print(f"  Sample surah:  {docs[0]['metadata']['surah']}")
        print(f"  Sample tafsir: {docs[0]['metadata']['tafsir_book']}")
        print(f"  Text length:   {len(docs[0]['text'])} chars")
    return docs

def process_quran_english():
    print("\nProcessing Quran English dataset...")
    path = "datasets/raw/quran/quran_english.json"

    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return []

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            surah_num   = str(entry.get("surah", ""))
            ayah_num    = str(entry.get("ayah", ""))
            surah_name  = clean_text(entry.get("surah-name-en", ""))
            ar_text     = clean_text(entry.get("arabic-text-uthmani", ""))
            en_text     = clean_text(entry.get("translation-en-itani", ""))
            surah_type  = entry.get("surah-type", "")

            if not en_text:
                continue

            ref = f"Surah {surah_num} ({surah_name}) Ayah {ayah_num}"

            docs.append({
                "id":          make_id("quran_en", i),
                "source_type": "quran",
                "text":        en_text,
                "arabic":      ar_text,
                "translation": en_text,
                "tafseer":     "",
                "metadata": {
                    "surah":      surah_num,
                    "ayah":       ayah_num,
                    "surah_name": surah_name,
                    "surah_type": surah_type,
                    "reference":  ref,
                    "grading":    "Authentic",
                    "language":   "en",
                }
            })

    out = "datasets/processed/quran_english_docs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  {len(docs):,} English Quran docs saved to {out}")
    if docs:
        print(f"  Sample ref:  {docs[0]['metadata']['reference']}")
        print(f"  Sample text: {docs[0]['text'][:100]}...")
    return docs

if __name__ == "__main__":
    hadith_docs      = process_hadith()
    quran_docs       = process_quran()          # Arabic tafseer
    quran_en_docs    = process_quran_english()  # English translations

    total = len(hadith_docs) + len(quran_docs) + len(quran_en_docs)
    print(f"\n{'='*50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Hadith docs        : {len(hadith_docs):,}")
    print(f"  Quran Arabic docs  : {len(quran_docs):,}")
    print(f"  Quran English docs : {len(quran_en_docs):,}")
    print(f"  Total              : {total:,}")
    print(f"{'='*50}")
    print("Next: python chunking/chunker.py")