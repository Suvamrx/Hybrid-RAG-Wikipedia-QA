import json
import os
import sys
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# =============================
# Sparse Retrieval (BM25) Script
# =============================
# This script builds a sparse BM25 index for Wikipedia chunks using tokenized text.
# It saves chunk metadata for retrieval and provides a sample retrieval function.

# Parameters for file paths
CHUNKS_PATH = os.path.join(os.getcwd(), 'data', 'wikipedia_chunks.json')
BM25_INDEX_PATH = os.path.join(os.getcwd(), 'data', 'bm25_index.json')  # (not used for saving index, but for consistency)
CHUNK_META_PATH = os.path.join(os.getcwd(), 'data', 'chunk_metadata.json')

# Load preprocessed Wikipedia chunks
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

texts = [chunk['text'] for chunk in chunks]
chunk_meta = [{k: v for k, v in chunk.items() if k != 'text'} for chunk in chunks]

# Tokenize the corpus for BM25
print('Tokenizing corpus...')
tokenized_corpus = [word_tokenize(text.lower()) for text in texts]

# Build a BM25 index for sparse retrieval
print('Building BM25 index...')
bm25 = BM25Okapi(tokenized_corpus)

# Example sparse retrieval function for interactive use
# Given a query, returns top_k most similar chunks with scores

def retrieve_sparse(query, top_k=5):
    # Tokenize and lowercase the query
    tokenized_query = word_tokenize(query.lower())
    # Get BM25 scores for the query against the corpus
    scores = bm25.get_scores(tokenized_query)
    # Retrieve the indices of the top_k highest scoring chunks
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    # Collect metadata and scores for the top chunks
    for idx in top_indices:
        meta = chunk_meta[idx].copy()
        meta['score'] = float(scores[idx])
        results.append(meta)
    return results

if __name__ == '__main__':
    # Example usage: interactive query
    if sys.stdin.isatty():
        # Interactive mode: allow user to query
        query = input('Enter your query: ')
        results = retrieve_sparse(query, top_k=5)
        for r in results:
            print(f"Score: {r['score']:.4f} | Title: {r['title']} | URL: {r['url']}")
    else:
        print("[INFO] Non-interactive mode detected. Skipping query input.")
