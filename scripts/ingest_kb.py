"""
ingest_kb.py — Chunk, embed, and load knowledge base documents into ChromaDB.

Run once before starting the server:
    python scripts/ingest_kb.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vector_db import LocalVectorDB
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
KB_DIR = ROOT / "knowledge_base"
DB_PATH = ROOT / "vector_db.json"
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 60        # character overlap between chunks
EMBED_MODEL = "models/gemini-embedding-2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            for sep in (". ", ".\n", "\n\n", "\n", " "):
                boundary = text.rfind(sep, start, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
    return chunks


def ingest():
    try:
        # Initialize pure Python local vector database
        db = LocalVectorDB(str(DB_PATH))
        # Clear existing data for idempotency
        db.delete_collection()

        # ── Process each knowledge base document ─────────────────────────────────
        docs = sorted(KB_DIR.glob("*.txt"))
        if not docs:
            log.error(f"No .txt files found in {KB_DIR}")
            sys.exit(1)

        all_ids, all_docs, all_embeddings, all_metas = [], [], [], []

        for doc_path in docs:
            log.info(f"Processing: {doc_path.name}")
            text = doc_path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            log.info(f"  → {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_path.stem}__chunk_{idx}"
                # Call Gemini embeddings API
                res = genai.embed_content(
                    model=EMBED_MODEL,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embedding = res["embedding"]

                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_embeddings.append(embedding)
                all_metas.append({"source": doc_path.name, "chunk_index": idx})

        # ── Batch upsert into LocalVectorDB ───────────────────────────────────────
        db.add(
            ids=all_ids,
            documents=all_docs,
            embeddings=all_embeddings,
            metadatas=all_metas,
        )

        log.info(f"\n✅ Ingested {len(all_ids)} chunks from {len(docs)} documents.")
        log.info(f"   LocalVectorDB stored at: {DB_PATH}")

        # ── Spot-check query ──────────────────────────────────────────────────────
        log.info("\n── Spot-check query: 'shipping time Canada' ──")
        q_res = genai.embed_content(
            model=EMBED_MODEL,
            content="shipping time Canada",
            task_type="retrieval_query"
        )
        q_embedding = q_res["embedding"]
        results = db.query(query_embeddings=[q_embedding], n_results=2)
        for i, doc in enumerate(results["documents"][0]):
            src = results["metadatas"][0][i]["source"]
            log.info(f"  [{src}] {doc[:120]}…")
    except Exception as e:
        log.exception("Ingestion failed with exception:")
        sys.exit(1)


if __name__ == "__main__":
    ingest()
