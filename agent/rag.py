"""
agent/rag.py — ChromaDB retrieval interface for BrightBox knowledge base.
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from agent.vector_db import LocalVectorDB
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "vector_db.json"
EMBED_MODEL = "models/gemini-embedding-2"

# Escalation trigger keywords — per doc3 escalation policy
ESCALATION_KEYWORDS = [
    "order number", "order #", "tracking number",
    "my account", "subscription id", "my billing",
    "frustrated", "angry", "unacceptable", "ridiculous",
    "exception", "special case", "override", "make an exception",
    "speak to a human", "speak to a person", "talk to someone",
    "real person", "manager", "supervisor",
]


@lru_cache(maxsize=1)
def _get_resources():
    """Lazily load the LocalVectorDB (singleton)."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {DB_PATH}. "
            "Run `python scripts/ingest_kb.py` first."
        )
    db = LocalVectorDB(str(DB_PATH))
    log.info(f"RAG loaded: {db.count()} chunks in LocalVectorDB")
    return db


def query_knowledge_base(user_query: str, n_results: int = 3) -> list[str]:
    """
    Embed user_query and return the top-n relevant text chunks from LocalVectorDB.
    Returns an empty list if the DB is not initialised.
    """
    try:
        db = _get_resources()
        # Call Gemini embeddings API
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=user_query,
            task_type="retrieval_query"
        )
        embedding = res["embedding"]
        results = db.query(query_embeddings=[embedding], n_results=n_results)
        chunks = results["documents"][0]  # list of strings
        log.debug(f"RAG returned {len(chunks)} chunks for: {user_query!r}")
        return chunks
    except FileNotFoundError as exc:
        log.warning(str(exc))
        return []
    except Exception as exc:
        log.error(f"RAG query failed: {exc}")
        return []


def should_escalate(user_query: str) -> bool:
    """
    Return True if the query matches escalation triggers defined in doc3.
    This is a fast keyword check — the LLM may also decide to escalate.
    """
    lower = user_query.lower()
    return any(kw in lower for kw in ESCALATION_KEYWORDS)
