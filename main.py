from fastapi import FastAPI

app = FastAPI(title="Islamic Knowledge RAG Assistant - Phase 1")

@app.get("/health")
def health():
    return {"status": "ok", "phase": 1}