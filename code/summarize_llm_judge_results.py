import json
import numpy as np

# Load LLM-judge results
with open('data/llm_judge_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract scores, ignoring nulls
factuality = [r['factuality'] for r in results if r['factuality'] is not None]
completeness = [r['completeness'] for r in results if r['completeness'] is not None]
relevance = [r['relevance'] for r in results if r['relevance'] is not None]

def summarize(scores, name):
    arr = np.array(scores)
    print(f"{name} - count: {len(arr)}")
    print(f"  Mean: {arr.mean():.2f}")
    print(f"  Median: {np.median(arr):.2f}")
    print(f"  Std: {arr.std():.2f}")
    print(f"  Min: {arr.min()}  Max: {arr.max()}")
    print()

print("LLM-Judge Results Summary:\n")
summarize(factuality, "Factuality")
summarize(completeness, "Completeness")
summarize(relevance, "Relevance")

# Count nulls
null_factuality = sum(1 for r in results if r['factuality'] is None)
null_completeness = sum(1 for r in results if r['completeness'] is None)
null_relevance = sum(1 for r in results if r['relevance'] is None)
print(f"Nulls - Factuality: {null_factuality}, Completeness: {null_completeness}, Relevance: {null_relevance}")
