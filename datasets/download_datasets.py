"""
datasets/download_datasets.py
-------------------------------
Downloads Islamic knowledge datasets from HuggingFace.
These become our verified internal knowledge base (Phase 1).

Run on Colab: !python datasets/download_datasets.py
"""

import os
import json
from datasets import load_dataset

# ── Folder setup ──────────────────────────────────────────
os.makedirs("datasets/raw/quran",   exist_ok=True)
os.makedirs("datasets/raw/hadith",  exist_ok=True)
os.makedirs("datasets/raw/tafseer", exist_ok=True)

# ══════════════════════════════════════════════════════════
# DATASET 1: Hadith — 6 major books, English + Arabic + grading
# Source: meeAtif/hadith_datasets
# Why: Has authenticity grading (Sahih/Daif) per hadith.
#      This directly powers our Truth Verification Mode.
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("DATASET 1: Hadith Collections")
print("=" * 60)

try:
    hadith_ds = load_dataset("meeAtif/hadith_datasets")

    print(f"Splits:        {list(hadith_ds.keys())}")
    print(f"Total entries: {len(hadith_ds['train'])}")
    print(f"Columns:       {hadith_ds['train'].column_names}")
    print("\nSample entry:")
    sample = hadith_ds['train'][0]
    for key, val in sample.items():
        print(f"  {key:30s}: {str(val)[:120]}")

    # Save raw
    out_path = "datasets/raw/hadith/hadith_all.json"
    hadith_ds['train'].to_json(out_path)
    print(f"\n✓ Saved → {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

except Exception as e:
    print(f"✗ Failed: {e}")


# ══════════════════════════════════════════════════════════
# DATASET 2: Quran with Tafseer (commentary)
# Source: MohamedRashad/Quran-Tafseer
# Why: Verse text + scholarly commentary = richer retrieval
#      than bare translation alone.
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DATASET 2: Quran + Tafseer")
print("=" * 60)

try:
    quran_ds = load_dataset("MohamedRashad/Quran-Tafseer")

    print(f"Splits:        {list(quran_ds.keys())}")
    print(f"Total entries: {len(quran_ds['train'])}")
    print(f"Columns:       {quran_ds['train'].column_names}")
    print("\nSample entry:")
    sample = quran_ds['train'][0]
    for key, val in sample.items():
        print(f"  {key:30s}: {str(val)[:150]}")

    # Save raw
    out_path = "datasets/raw/quran/quran_tafseer.json"
    quran_ds['train'].to_json(out_path)
    print(f"\n✓ Saved → {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

except Exception as e:
    print(f"✗ Failed: {e}")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for root, dirs, files in os.walk("datasets/raw"):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {path:55s} {size:.1f} MB")

print("\nDataset download complete.")
print("Next step: preprocessing/cleaner.py")