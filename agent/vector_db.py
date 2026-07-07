import json
import os
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

class LocalVectorDB:
    """
    A pure-Python local vector database storing embeddings in a JSON file
    and calculating cosine similarity via NumPy. Bypasses native C++ dependencies.
    """
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.data = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
        self.load()

    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                log.info(f"Loaded vector DB from {self.db_path} containing {len(self.data['ids'])} items.")
            except Exception as e:
                log.warning(f"Failed to load vector DB: {e}")

    def save(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Failed to save vector DB: {e}")

    def add(self, ids, documents, embeddings, metadatas):
        """Mock behavior of chroma collection.add"""
        for i, doc_id in enumerate(ids):
            if doc_id in self.data["ids"]:
                idx = self.data["ids"].index(doc_id)
                self.data["documents"][idx] = documents[i]
                self.data["embeddings"][idx] = embeddings[i]
                self.data["metadatas"][idx] = metadatas[i]
            else:
                self.data["ids"].append(doc_id)
                self.data["documents"].append(documents[i])
                self.data["embeddings"].append(embeddings[i])
                self.data["metadatas"].append(metadatas[i])
        self.save()

    def count(self) -> int:
        return len(self.data["ids"])

    def delete_collection(self):
        """Reset database state and remove storage file if it exists."""
        if self.db_path.exists():
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        self.data = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    def query(self, query_embeddings, n_results=3):
        """Mock behavior of chroma collection.query"""
        if not self.data["embeddings"] or not query_embeddings:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        # We take the first query embedding (query_embeddings[0]) to match Chroma's signature
        q = np.array(query_embeddings[0])
        norms_q = np.linalg.norm(q)
        if norms_q == 0:
            norms_q = 1e-9

        similarities = []
        for emb in self.data["embeddings"]:
            e = np.array(emb)
            norm_e = np.linalg.norm(e)
            if norm_e == 0:
                norm_e = 1e-9
            sim = np.dot(q, e) / (norms_q * norm_e)
            similarities.append(float(sim))

        # Sort indices descending (highest similarity first)
        sorted_indices = np.argsort(similarities)[::-1][:n_results]

        res_docs = [self.data["documents"][idx] for idx in sorted_indices]
        res_metas = [self.data["metadatas"][idx] for idx in sorted_indices]
        res_ids = [self.data["ids"][idx] for idx in sorted_indices]

        return {
            "documents": [res_docs],
            "metadatas": [res_metas],
            "ids": [res_ids],
            "distances": [[1.0 - similarities[idx] for idx in sorted_indices]]
        }
