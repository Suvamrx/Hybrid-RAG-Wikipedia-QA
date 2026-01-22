import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys

# =============================
# Dense Retrieval (FAISS) Script
# =============================
# This script builds a dense vector index (FAISS) for Wikipedia chunks using a SentenceTransformer model.
# It saves the index and chunk metadata for fast retrieval, and provides a sample retrieval function.

# Parameters for embedding model and file paths
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNKS_PATH = os.path.join(os.getcwd(), 'data', 'wikipedia_chunks.json')
FAISS_INDEX_PATH = os.path.join(os.getcwd(), 'data', 'faiss_index.bin')
CHUNK_META_PATH = os.path.join(os.getcwd(), 'data', 'chunk_metadata.json')

# Load preprocessed Wikipedia chunks
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

texts = [chunk['text'] for chunk in chunks]

# Load the sentence embedding model
print('Loading embedding model...')
model = SentenceTransformer(EMBEDDING_MODEL)

# Compute dense embeddings for all chunks
print('Computing embeddings...')
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

# Build a FAISS index for fast dense retrieval (cosine similarity)
print('Building FAISS index...')
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Cosine similarity (with normalized vectors)
index.add(embeddings)

# Save the FAISS index to disk
faiss.write_index(index, FAISS_INDEX_PATH)
print(f'FAISS index saved to {FAISS_INDEX_PATH}')

# Save chunk metadata (without text) for retrieval and display
chunk_meta = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]
with open(CHUNK_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(chunk_meta, f, indent=2, ensure_ascii=False)
print(f'Chunk metadata saved to {CHUNK_META_PATH}')

# Example dense retrieval function for interactive use
# Given a query, returns top_k most similar chunks with scores

def retrieve_dense(query, top_k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype('float32'), top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        meta = chunk_meta[idx]
        meta['score'] = float(score)
        results.append(meta)
    return results

if __name__ == '__main__':
    # Example usage: interactive query
    if sys.stdin.isatty():
        # Interactive mode: allow user to query
        query = input('Enter your query: ')
        results = retrieve_dense(query, top_k=5)
        for r in results:
            print(f"Score: {r['score']:.4f} | Title: {r['title']} | URL: {r['url']}")
    else:
        print("[INFO] Non-interactive mode detected. Skipping query input.")
