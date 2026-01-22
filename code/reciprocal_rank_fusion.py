import json
import os
import sys
from dense_retrieval_faiss import retrieve_dense
from sparse_retrieval_bm25 import retrieve_sparse

# =====================================
# Reciprocal Rank Fusion (RRF) Script
# =====================================
# This script fuses dense and sparse retrieval results using the RRF algorithm.
# It loads chunk metadata, combines results, and provides a sample interactive query.

# Load all chunk text for lookup (for display in fused results)
CHUNKS_PATH = os.path.join(os.getcwd(), 'data', 'wikipedia_chunks.json')
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    _all_chunks = json.load(f)
_chunkid_to_text = {c['chunk_id']: c['text'] for c in _all_chunks}

# Reciprocal Rank Fusion algorithm
# Combines ranked lists from dense and sparse retrievals into a single fused ranking
# k: RRF constant (controls score decay), top_n: number of results to return

def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=5):
    # Create a set of all unique chunk IDs from both dense and sparse results
    all_ids = set([r['chunk_id'] for r in dense_results] + [r['chunk_id'] for r in sparse_results])
    scores = {}
    for chunk_id in all_ids:
        # Find the rank of the chunk in both dense and sparse results
        rank_dense = next((i for i, r in enumerate(dense_results) if r['chunk_id'] == chunk_id), None)
        rank_sparse = next((i for i, r in enumerate(sparse_results) if r['chunk_id'] == chunk_id), None)
        score = 0.0
        # Calculate score contribution from dense result rank
        if rank_dense is not None:
            score += 1.0 / (k + rank_dense + 1)
        # Calculate score contribution from sparse result rank
        if rank_sparse is not None:
            score += 1.0 / (k + rank_sparse + 1)
        scores[chunk_id] = score
    # Sort all chunk IDs by their RRF score in descending order and take the top_n
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    # Merge metadata for output
    id_to_meta = {r['chunk_id']: r for r in dense_results + sparse_results}
    fused_results = []
    for cid in sorted_ids:
        meta = id_to_meta[cid].copy()
        meta['rrf_score'] = scores[cid]
        # Add text field from original chunk
        meta['text'] = _chunkid_to_text.get(cid, '')
        fused_results.append(meta)
    return fused_results

if __name__ == '__main__':
    # Example usage: interactive query
    if sys.stdin.isatty():
        # Interactive mode: allow user to query
        query = input('Enter your query: ')
        dense_results = retrieve_dense(query, top_k=20)
        sparse_results = retrieve_sparse(query, top_k=20)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=5)
        print('\nTop results by Reciprocal Rank Fusion:')
        for r in fused:
            print(f"RRF Score: {r['rrf_score']:.4f} | Title: {r['title']} | URL: {r['url']}")
    else:
        print("[INFO] Non-interactive mode detected. Skipping query input.")