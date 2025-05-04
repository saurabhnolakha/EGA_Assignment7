import os
from pathlib import Path
import faiss
import numpy as np
import requests
import json

# -- CONFIG --
CHUNK_SIZE = 50
CHUNK_OVERLAP = 15
TEST_DIR = Path(__file__).parent / "test"
TEST_DIR.mkdir(exist_ok=True)
INDEX_FILE = Path(__file__).parent / "faiss_index_ollama.bin"
METADATA_FILE = Path(__file__).parent / "metadata.json"
OLLAMA_METADATA_FILE = Path(__file__).parent / "metadata_ollama.json"
EXTRACTED_DIR = Path(__file__).parent / "extracted_content"
EXTRACTED_DIR.mkdir(exist_ok=True)

# -- HELPERS --
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

# -- MAIN LOGIC --
# Find all .md files in extracted_content directory
md_files = list(EXTRACTED_DIR.glob("*.md"))
print("Markdown files found:", [f.name for f in md_files])

# Load or initialize FAISS index and Ollama metadata
if INDEX_FILE.exists():
    print(f"Loading existing FAISS index from {INDEX_FILE}")
    index = faiss.read_index(str(INDEX_FILE))
else:
    index = None

if OLLAMA_METADATA_FILE.exists():
    with open(OLLAMA_METADATA_FILE, "r", encoding="utf-8") as f:
        ollama_metadata = json.load(f)
else:
    ollama_metadata = []

# For fast lookup of already indexed files
ollama_indexed_docs = set(m.get("doc_name") for m in ollama_metadata)

for file in md_files:
    if file.name in ollama_indexed_docs:
        print(f"Skipping already Ollama-indexed file: {file.name}")
        continue
    print(f"Processing: {file.name}")
    # Read the corresponding .url file
    url_file = file.with_suffix('.url')
    if url_file.exists():
        with open(url_file, 'r', encoding='utf-8') as uf:
            url = uf.read().strip()
    else:
        url = "UNKNOWN"
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = chunk_text(content)
        print(f"Chunks for {file.name}: {len(chunks)}")
        embeddings = []
        new_metadata = []
        for idx, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            embeddings.append(emb)
            new_metadata.append({
                "doc_name": file.name,
                "url": url,
                "chunk": chunk,
                "chunk_id": f"{file.stem}_{idx}"
            })
            print(f"Getting embedding for chunk {idx+1}/{len(chunks)} (length: {len(chunk)})")
        if embeddings:
            embeddings_np = np.stack(embeddings)
            if index is None:
                dim = len(embeddings[0])
                index = faiss.IndexFlatL2(dim)
            index.add(embeddings_np)
            ollama_metadata.extend(new_metadata)
            print(f"Added {len(embeddings)} embeddings from {file.name}")

# Save updated index and Ollama metadata
if index is not None and index.ntotal > 0:
    faiss.write_index(index, str(INDEX_FILE))
    with open(OLLAMA_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(ollama_metadata, f, indent=2)
    print(f"✅ Ollama FAISS index and metadata_ollama.json updated. Total vectors: {index.ntotal}")
else:
    print("No new data to index.") 