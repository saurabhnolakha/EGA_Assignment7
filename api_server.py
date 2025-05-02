from flask import Flask, request, jsonify
import faiss
import numpy as np
import json
from pathlib import Path
from embedding_generator import get_embedding as get_embedding_gemini
from embedding_generator_ollama import get_embedding as get_embedding_ollama

app = Flask(__name__)

# File paths for both models
GEMINI_INDEX_FILE = Path(__file__).parent / "faiss_index.bin"
GEMINI_METADATA_FILE = Path(__file__).parent / "metadata.json"
OLLAMA_INDEX_FILE = Path(__file__).parent / "faiss_index_ollama.bin"
OLLAMA_METADATA_FILE = Path(__file__).parent / "metadata_ollama.json"

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    k = int(data.get("k", 3))
    model = data.get("model", "gemini").lower()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if model == "ollama":
        index_file = OLLAMA_INDEX_FILE
        metadata_file = OLLAMA_METADATA_FILE
        embed_fn = get_embedding_ollama
    else:
        index_file = GEMINI_INDEX_FILE
        metadata_file = GEMINI_METADATA_FILE
        embed_fn = get_embedding_gemini

    if not index_file.exists() or not metadata_file.exists():
        return jsonify({"error": f"Index or metadata file for model '{model}' not found."}), 404

    # Load index and metadata
    index = faiss.read_index(str(index_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Get embedding for query
    query_vec = embed_fn(query).reshape(1, -1)
    print(f"[{model}] Index dimension:", index.d)
    print(f"[{model}] Query vector shape:", query_vec.shape)
    D, I = index.search(query_vec, k)

    # Prepare results
    results = []
    for idx, dist in zip(I[0], D[0]):
        data = metadata[idx]
        results.append({
            "doc_name": data["doc_name"],
            "url": data["url"],
            "chunk": data["chunk"],
            "chunk_id": data["chunk_id"],
            "score": float(dist)
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000) 