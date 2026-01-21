# Sparse Only Ablation
import json
import os
from generate_response_llm import build_context, generate_answer
from reciprocal_rank_fusion import retrieve_sparse
from rouge_score import rouge_scorer
import numpy as np


QA_PATH = os.path.join(os.getcwd(), 'data', 'generated_qa_pairs.json')
RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'evaluation_results_sparse.json')
TOP_K = 10

with open(QA_PATH, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)
# Load all chunks for text lookup
with open('data/wikipedia_chunks.json', 'r', encoding='utf-8') as f_chunks:
    all_chunks = json.load(f_chunks)
chunkid_to_text = {str(c['chunk_id']): c['text'] for c in all_chunks}

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

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

def compute_mrr(gt_url, retrieved_chunks):
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if chunk['url'] == gt_url:
            return 1.0 / rank
    return 0.0

results = []
for i, qa in enumerate(qa_pairs):
    print(f"Evaluating Q{i+1}/{len(qa_pairs)}: {qa['question']}")
    sparse_results = retrieve_sparse(qa['question'], top_k=TOP_K)
    for r in sparse_results:
        r['text'] = chunkid_to_text.get(str(r['chunk_id']), '')
    context = "\n\n".join([f"Source: {r['title']}\n{r['text']}" for r in sparse_results])
    pred_answer = generate_answer(qa['question'], context)
    mrr = compute_mrr(qa['source_url'], sparse_results)
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

with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

mrrs = [r['mrr'] for r in results]
f1s = [r['f1'] for r in results]
rouges = [r['rougeL'] for r in results]
print(f"\nMean Reciprocal Rank (MRR): {np.mean(mrrs):.4f}")
print(f"Mean F1 Score: {np.mean(f1s):.4f}")
print(f"Mean ROUGE-L: {np.mean(rouges):.4f}")
print(f"Results saved to {RESULTS_PATH}")
