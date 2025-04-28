import os
from pathlib import Path
import faiss
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
import hashlib

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
BATCH_SIZE = 10  # Number of chunks to process in one API call

# Reused functions from faiss_advanced.py
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks with specified size and overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding_gemini(texts: list) -> list:
    """Get embeddings using Gemini model for a batch of texts."""
    try:
        res = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [np.array(embedding.values, dtype=np.float32) for embedding in res.embeddings]
    except Exception as e:
        print(f"Error in batch embedding: {str(e)}")
        # If batch fails, try processing texts individually
        embeddings = []
        for text in texts:
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings.append(np.array(res.embeddings[0].values, dtype=np.float32))
            except Exception as e:
                print(f"Error processing individual text: {str(e)}")
                # Add zero vector for failed embeddings
                embeddings.append(np.zeros(768, dtype=np.float32))
        return embeddings

def get_embedding_ollama(texts: list) -> list:
    """Get embeddings using Ollama for a batch of texts."""
    import requests
    embeddings = []
    for text in texts:
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text}
            )
            embeddings.append(np.array(response.json()["embedding"], dtype=np.float32))
        except Exception as e:
            print(f"Error processing text with Ollama: {str(e)}")
            embeddings.append(np.zeros(768, dtype=np.float32))
    return embeddings

# Function to choose between embedding models
def get_embedding(texts: list, model="gemini") -> list:
    """Get embeddings using specified model for a batch of texts."""
    if model == "gemini":
        return get_embedding_gemini(texts)
    elif model == "ollama":
        return get_embedding_ollama(texts)
    else:
        raise ValueError(f"Unsupported model: {model}")

# Function to process new files
def process_new_files(directory_path, index_file="faiss_index.bin", metadata_file="metadata.json", 
                      cache_file="processed_files.json", model="gemini"):
    """Process new files and update the FAISS index."""
    directory_path = Path(directory_path)
    
    # Load cache of processed files
    processed_files = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            processed_files = json.load(f)
    
    # Load existing metadata if available
    metadata = []
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Load existing index if available
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print(f"Loaded existing index with {index.ntotal} vectors")
    else:
        # We'll create index later when we have the dimension
        index = None
    
    # Process markdown files
    new_chunks = []
    new_metadata = []
    
    for file in directory_path.glob("*.md"):
        file_path = str(file)
        file_hash = None
        
        # Calculate file hash to detect changes
        with open(file, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Skip if file hasn't changed
        if file_path in processed_files and processed_files[file_path] == file_hash:
            print(f"Skipping unchanged file: {file.name}")
            continue
        
        print(f"Processing file: {file.name}")
        
        # Process the file
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = chunk_text(content)
            
            # Process chunks in batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                print(f"Processing batch of {len(batch)} chunks...")
                
                # Get embeddings for the batch
                embeddings = get_embedding(batch, model)
                
                # Add to new chunks and metadata
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    new_chunks.append(embedding)
                    new_metadata.append({
                        "doc_name": file.name,
                        "chunk": chunk,
                        "chunk_id": f"{file.stem}_{i+idx}"
                    })
                
                # Respect rate limits
                if model == "gemini":
                    print(f"Rate limiting: sleeping for 12 seconds")
                    time.sleep(12)  # Stay below 5 RPM
        
        # Update processed files cache
        processed_files[file_path] = file_hash
    
    # Create or update index
    if new_chunks:
        if index is None:
            # Create new index with the right dimension
            dimension = len(new_chunks[0])
            index = faiss.IndexFlatL2(dimension)
            print(f"Created new index with dimension {dimension}")
        
        # Add new vectors
        index.add(np.stack(new_chunks))
        print(f"Added {len(new_chunks)} new vectors to index")
        
        # Update metadata
        metadata.extend(new_metadata)
        
        # Save index
        faiss.write_index(index, index_file)
        print(f"Saved index to {index_file}")
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        print(f"Saved metadata to {metadata_file}")
        
        # Save processed files cache
        with open(cache_file, 'w') as f:
            json.dump(processed_files, f)
        print(f"Updated processed files cache")
    else:
        print("No new files to process")
    
    return index, metadata

if __name__ == "__main__":
    # Example usage
    index, metadata = process_new_files("./test", model="gemini")
    print(f"Index now has {index.ntotal} total vectors")
    
    # Example search
    query = "What is Natural Language Processing?"
    # Get the embedding and ensure correct shape for FAISS
    query_embedding = get_embedding([query], model="gemini")[0]  # Get the first (and only) embedding
    query_vec = np.array([query_embedding])  # Create 2D array with shape (1, d)
    
    k = 3  # number of results
    D, I = index.search(query_vec, k=k)

    print(f"\n🔍 Query: {query}\n\n📚 Top {k} Matches:")
    for rank, idx in enumerate(I[0]):
        if idx < len(metadata):  # Safety check
            data = metadata[idx]
            print(f"\n#{rank + 1}: From {data['doc_name']} [{data['chunk_id']}]")
            print(f"→ {data['chunk']}") 