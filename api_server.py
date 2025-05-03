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

app = Flask(__name__)
CORS(app)

# File paths for both models
GEMINI_INDEX_FILE = Path(__file__).parent / "faiss_index.bin"
GEMINI_METADATA_FILE = Path(__file__).parent / "metadata.json"
OLLAMA_INDEX_FILE = Path(__file__).parent / "faiss_index_ollama.bin"
OLLAMA_METADATA_FILE = Path(__file__).parent / "metadata_ollama.json"
EXTRACTED_DIR = Path(__file__).parent / "extracted_content"
EXTRACTED_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

@app.route("/add", methods=["POST"])
def add_content():
    data = request.json
    url = data.get("url")
    title = data.get("title")
    timestamp = data.get("timestamp", int(time.time()))

    if not url:
        return jsonify({"error": "Missing url"}), 400

    # Check if URL is already indexed in metadata_ollama.json
    if OLLAMA_METADATA_FILE.exists():
        with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if any(m.get("url") == url for m in metadata):
            return jsonify({"skipped": True, "reason": "URL already indexed"}), 200
    else:
        metadata = []

    # Fetch the page content
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        html_content = resp.text
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

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

    return jsonify({"success": True, "filename": filename, "chunks": len(chunks)})

if __name__ == "__main__":
    app.run(debug=True, port=5000) 