
from pathlib import Path
import json

import faiss



ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = ROOT / "vector_store"

INDEX_FILE = STORE_DIR / "faiss.index"
DOCUMENTS_FILE = STORE_DIR / "documents.json"

_model = None
_index = None
_documents = None


def load_vector_store():
    from sentence_transformers import SentenceTransformer
    """Load the embedding model, FAISS index and indexed documents."""
    global _model, _index, _documents

    if _model is None:
        if not INDEX_FILE.exists() or not DOCUMENTS_FILE.exists():
            raise FileNotFoundError(
                "FAISS files are missing. Run: "
                "python rag/build_vector_store.py"
            )

        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        _index = faiss.read_index(str(INDEX_FILE))
        _documents = json.loads(
            DOCUMENTS_FILE.read_text(encoding="utf-8")
        )

    return _model, _index, _documents


def retrieve_documents(query: str, k: int = 2) -> list[dict]:
    """Retrieve the most relevant FAQ documents for a query."""
    model, index, documents = load_vector_store()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(k, len(documents))

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, document_index in zip(scores[0], indices[0]):
        if document_index < 0:
            continue

        results.append({
            "text": documents[int(document_index)],
            "score": float(score),
        })

    return results
