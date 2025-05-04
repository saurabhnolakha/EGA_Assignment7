from flask import Flask, request, jsonify
import faiss
import numpy as np
import json
from pathlib import Path
#from embedding_generator import get_embedding as get_embedding_gemini
from embedding_generator_ollama import get_embedding as get_embedding_ollama
import time
import requests
from content_extractor import extract_content
from flask_cors import CORS
from filelock import FileLock, Timeout

app = Flask(__name__)
CORS(app)

# File paths for both models
GEMINI_INDEX_FILE = Path(__file__).parent / "faiss_index.bin"
GEMINI_METADATA_FILE = Path(__file__).parent / "metadata.json"
OLLAMA_INDEX_FILE = Path(__file__).parent / "faiss_index_ollama.bin"
OLLAMA_METADATA_FILE = Path(__file__).parent / "metadata_ollama.json"
EXTRACTED_DIR = Path(__file__).parent / "extracted_content"
EXTRACTED_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 50
CHUNK_OVERLAP = 15
LOCK_FILE = "metadata_ollama.lock"

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

@app.route("/add", methods=["POST"])
def add_url():
    data = request.get_json()
    url = data.get("url")
    title = data.get("title")
    timestamp = data.get("timestamp")
    # Normalize URL if needed

    # First lock: check and update metadata (add as pending)
    try:
        with FileLock(LOCK_FILE, timeout=60):
            if OLLAMA_METADATA_FILE.exists():
                with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = []
            # Check for duplicate
            if any(m.get("url") == url for m in metadata):
                print(f"[EGA API] URL already indexed, skipping: {url}")
                return jsonify({"skipped": True, "reason": "URL already indexed"}), 200
            # Add as pending
            pending_entry = {
                "url": url,
                "title": title,
                "timestamp": timestamp,
                "status": "pending"
            }
            metadata.append(pending_entry)
            with open(OLLAMA_METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Timeout:
        return jsonify({"error": "Server busy, please try again later."}), 503

    # Do extraction, chunking, embedding outside the lock
    try:
        # Fetch the page content
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        html_content = resp.text

        # Extract main content
        markdown_content = extract_content(html_content, url)
        if not markdown_content:
            return jsonify({"error": "Extraction failed or content too short or domain not allowed."}), 400

        # Save content and url
        safe_title = "_".join((title or "page").split())[:40]
        filename = f"extracted_content_{safe_title}_{timestamp}.md"
        file_path = EXTRACTED_DIR / filename
        url_path = file_path.with_suffix('.url')
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        with open(url_path, "w", encoding="utf-8") as f:
            f.write(url)

        # Chunk, embed, and update Ollama index and metadata
        chunks = chunk_text(markdown_content)
        if not chunks:
            return jsonify({"error": "No chunks generated from content."}), 400

        # Load or initialize index and metadata
        if OLLAMA_INDEX_FILE.exists():
            index = faiss.read_index(str(OLLAMA_INDEX_FILE))
        else:
            index = None

        embeddings = []
        new_metadata = []
        for idx, chunk in enumerate(chunks):
            emb = get_embedding_ollama(chunk)
            embeddings.append(emb)
            new_metadata.append({
                "doc_name": filename,
                "url": url,
                "chunk": chunk,
                "chunk_id": f"{Path(filename).stem}_{idx}"
            })
        if embeddings:
            embeddings_np = np.stack(embeddings)
            if index is None:
                dim = len(embeddings[0])
                index = faiss.IndexFlatL2(dim)
            index.add(embeddings_np)
            metadata.extend(new_metadata)
            faiss.write_index(index, str(OLLAMA_INDEX_FILE))
            with open(OLLAMA_METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"[EGA API] Indexed and embedded {len(embeddings)} chunks from {filename}")
        else:
            print(f"[EGA API] No embeddings generated for {filename}")

        # Second lock: update metadata entry to 'done'
        with FileLock(LOCK_FILE, timeout=60):
            if OLLAMA_METADATA_FILE.exists():
                with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                for m in metadata:
                    if m.get("url") == url:
                        m["status"] = "done"
                with open(OLLAMA_METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

        return jsonify({"success": True, "url": url}), 200
    except Exception as e:
        # If processing fails, remove the pending entry
        with FileLock(LOCK_FILE, timeout=60):
            if OLLAMA_METADATA_FILE.exists():
                with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata = [m for m in metadata if m.get("url") != url]
                with open(OLLAMA_METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    #k = int(data.get("k", 5))
    k=10
    model = data.get("model", "ollama")
    rerank = data.get("rerank", True)
    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Use Ollama index and metadata
    if not OLLAMA_INDEX_FILE.exists() or not OLLAMA_METADATA_FILE.exists():
        return jsonify({"error": "Ollama index or metadata not found."}), 404

    # Load index and metadata
    index = faiss.read_index(str(OLLAMA_INDEX_FILE))
    with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Get embedding for query
    query_vec = get_embedding_ollama(query).reshape(1, -1)
    D, I = index.search(query_vec, k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(metadata[idx])
    if rerank:
        # Re-rank so that exact/substring matches come first
        query_lower = query.lower()
        exact_matches = [r for r in results if query_lower in r.get('chunk', '').lower()]
        non_exact_matches = [r for r in results if r not in exact_matches]
        reranked_results = exact_matches + non_exact_matches
        # Print top 3 results to server console
        print("[EGA API] Top 3 search results for query (after re-ranking):", query)
        for i, r in enumerate(reranked_results[:3]):
            print(f"Result {i+1}: Title: {r.get('title', r.get('doc_name', 'Untitled'))}, URL: {r.get('url')}, Snippet: {r.get('chunk', '')[:100]}")
        return jsonify({"results": reranked_results[:k]})
    else:
        # No re-ranking, just return FAISS results
        print("[EGA API] Top 3 search results for query (semantic only):", query)
        for i, r in enumerate(results[:3]):
            print(f"Result {i+1}: Title: {r.get('title', r.get('doc_name', 'Untitled'))}, URL: {r.get('url')}, Snippet: {r.get('chunk', '')[:100]}")
        return jsonify({"results": results[:k]})

if __name__ == "__main__":
    app.run(debug=True, port=5000) 