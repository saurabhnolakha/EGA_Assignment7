import os
from pathlib import Path
import faiss
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -- CONFIG --
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
TEST_DIR = Path(__file__).parent / "test"
TEST_DIR.mkdir(exist_ok=True)
INDEX_FILE = Path(__file__).parent  / "faiss_index.bin"
METADATA_FILE = Path(__file__).parent / "metadata.json"
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
    res = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return np.array(res.embeddings[0].values, dtype=np.float32)

# -- MAIN LOGIC --
# Find all .md files in extracted_content directory
md_files = list(EXTRACTED_DIR.glob("*.md"))
print("Markdown files found:", [f.name for f in md_files])

# Load or initialize FAISS index and metadata
if INDEX_FILE.exists():
    print(f"Loading existing FAISS index from {INDEX_FILE}")
    index = faiss.read_index(str(INDEX_FILE))
else:
    index = None

if METADATA_FILE.exists():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = []

for file in md_files:
    # Check if this file is already indexed (by filename in metadata)
    already_indexed = any(m.get("doc_name") == file.name for m in metadata)
    if already_indexed:
        print(f"Skipping already indexed file: {file.name}")
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
            print(f"Sleeping for 12 seconds to respect API limits...")
            time.sleep(12)
        if embeddings:
            embeddings_np = np.stack(embeddings)
            if index is None:
                dim = len(embeddings[0])
                index = faiss.IndexFlatL2(dim)
            index.add(embeddings_np)
            metadata.extend(new_metadata)
            print(f"Added {len(embeddings)} embeddings from {file.name}")

# Save updated index and metadata
if index is not None and index.ntotal > 0:
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ FAISS index and metadata updated. Total vectors: {index.ntotal}")
else:
    print("No new data to index.")
