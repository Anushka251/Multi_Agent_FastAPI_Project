from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "faq.txt"
STORE_DIR = ROOT / "vector_store"

STORE_DIR.mkdir(parents=True, exist_ok=True)

# Load the sample knowledge base
text = DATA_FILE.read_text(encoding="utf-8")

# Split the sample file into individual documents
documents = [
    part.strip()
    for part in text.split("\n\n")
    if part.strip()
]

if not documents:
    raise ValueError("No documents found in data/faq.txt")

print("Documents loaded:", len(documents))

# Generate local text embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

# Create a FAISS similarity index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save the index and document text
faiss.write_index(index, str(STORE_DIR / "faiss.index"))

(STORE_DIR / "documents.json").write_text(
    json.dumps(documents, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("FAISS vector store created successfully.")
print("Documents indexed:", index.ntotal)
print("Embedding dimensions:", embeddings.shape[1])
print("Saved to:", STORE_DIR)