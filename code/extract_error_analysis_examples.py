import json

# Load LLM-judge results
with open('data/llm_judge_results.json', 'r', encoding='utf-8') as f:
    judge_results = json.load(f)

# Load evaluation results (hybrid)
with open('data/evaluation_results.json', 'r', encoding='utf-8') as f:
    eval_results = json.load(f)

# Find low-scoring LLM-judge answers (factuality, completeness, or relevance <= 2)
low_judge = [
    r for r in judge_results if (
        (r.get('factuality') is not None and r['factuality'] <= 2)
        or (r.get('completeness') is not None and r['completeness'] <= 2)
        or (r.get('relevance') is not None and r['relevance'] <= 2)
    )
]

# Map question_id to evaluation result for context
id2eval = {r['question_id']: r for r in eval_results}

print("Low-scoring LLM-Judge Examples (factuality, completeness, or relevance <= 2):\n")
for r in low_judge[:5]:  # Show up to 5 examples
    qid = r['question_id']
    eval_r = id2eval.get(qid, {})
    print(f"Question ID: {qid}")
    print(f"Factuality: {r.get('factuality')}, Completeness: {r.get('completeness')}, Relevance: {r.get('relevance')}")
    print(f"Explanation: {r.get('explanation')}")
    print(f"Question: {eval_r.get('question')}")
    print(f"Ground Truth: {eval_r.get('ground_truth')}")
    print(f"Generated Answer: {eval_r.get('generated_answer')}")
    print(f"MRR: {eval_r.get('mrr')}, F1: {eval_r.get('f1')}, ROUGE-L: {eval_r.get('rougeL')}")
    print('-'*80)
