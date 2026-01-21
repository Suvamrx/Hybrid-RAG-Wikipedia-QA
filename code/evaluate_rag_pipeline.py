import json
import os
from generate_response_llm import build_context, generate_answer
from reciprocal_rank_fusion import reciprocal_rank_fusion, retrieve_dense, retrieve_sparse
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import numpy as np

# =====================================
# RAG Evaluation Pipeline Script
# =====================================
# This script evaluates the RAG system on a set of Q&A pairs.
# It computes MRR, F1, and ROUGE-L metrics for each question, saves results, and prints a summary.

# File paths and evaluation parameters
QA_PATH = os.path.join(os.getcwd(), 'data', 'generated_qa_pairs.json')
RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'evaluation_results.json')
TOP_K = 10  # Number of top chunks to retrieve for evaluation

# Load generated Q&A pairs for evaluation
with open(QA_PATH, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Compute F1 score between predicted and ground truth answers (token overlap)
def compute_f1(pred, gt):
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not pred_tokens or not gt_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Compute Mean Reciprocal Rank (MRR) for retrieval
# Returns 1/rank if the correct source URL is in the top-K, else 0
def compute_mrr(gt_url, retrieved_chunks):
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if chunk['url'] == gt_url:
            return 1.0 / rank
    return 0.0

results = []
for i, qa in enumerate(qa_pairs):
    print(f"Evaluating Q{i+1}/{len(qa_pairs)}: {qa['question']}")
    # Retrieve top-K chunks using dense, sparse, and RRF fusion
    dense_results = retrieve_dense(qa['question'], top_k=TOP_K)
    sparse_results = retrieve_sparse(qa['question'], top_k=TOP_K)
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=TOP_K)
    # Generate answer using the LLM and retrieved context
    context = "\n\n".join([f"Source: {r['title']}\n{r['text']}" for r in fused])
    pred_answer = generate_answer(qa['question'], context)
    # Compute evaluation metrics
    mrr = compute_mrr(qa['source_url'], fused)
    f1 = compute_f1(pred_answer, qa['answer'])
    rouge = scorer.score(qa['answer'], pred_answer)['rougeL'].fmeasure
    results.append({
        'question_id': qa['question_id'],
        'question': qa['question'],
        'ground_truth': qa['answer'],
        'generated_answer': pred_answer,
        'source_url': qa['source_url'],
        'mrr': mrr,
        'f1': f1,
        'rougeL': rouge,
    })

# Save evaluation results to JSON for reporting
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print summary statistics for all metrics
mrrs = [r['mrr'] for r in results]
f1s = [r['f1'] for r in results]
rouges = [r['rougeL'] for r in results]
print(f"\nMean Reciprocal Rank (MRR): {np.mean(mrrs):.4f}")
print(f"Mean F1 Score: {np.mean(f1s):.4f}")
print(f"Mean ROUGE-L: {np.mean(rouges):.4f}")
print(f"Results saved to {RESULTS_PATH}")
