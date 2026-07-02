# configs/paths.py
# Central path config — change STORAGE_ROOT to switch between local and Drive

import os

# On Colab with Drive mounted → /content/drive/MyDrive/Seerah-RAG-Data
# On local Windows             → datasets  (relative)
STORAGE_ROOT = os.environ.get("SEERAH_STORAGE", "datasets")

PATHS = {
    "raw_hadith":       os.path.join(STORAGE_ROOT, "raw", "hadith"),
    "raw_quran":        os.path.join(STORAGE_ROOT, "raw", "quran"),
    "processed":        os.path.join(STORAGE_ROOT, "processed"),
    "chunks":           os.path.join(STORAGE_ROOT, "chunks"),
    "vector_db":        os.path.join(STORAGE_ROOT, "vector_db", "chroma"),
}

def make_all_dirs():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)