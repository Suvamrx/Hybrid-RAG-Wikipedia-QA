import json
import matplotlib.pyplot as plt
import numpy as np

with open('data/llm_judge_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

factuality = [r['factuality'] for r in results if r['factuality'] is not None]
completeness = [r['completeness'] for r in results if r['completeness'] is not None]
relevance = [r['relevance'] for r in results if r['relevance'] is not None]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(factuality, bins=np.arange(1, 7)-0.5, edgecolor='black', color='skyblue')
plt.title('Factuality')
plt.xlabel('Score')
plt.ylabel('Count')
plt.xticks(range(1, 6))

plt.subplot(1, 3, 2)
plt.hist(completeness, bins=np.arange(1, 7)-0.5, edgecolor='black', color='lightgreen')
plt.title('Completeness')
plt.xlabel('Score')
plt.ylabel('Count')
plt.xticks(range(1, 6))

plt.subplot(1, 3, 3)
plt.hist(relevance, bins=np.arange(1, 7)-0.5, edgecolor='black', color='salmon')
plt.title('Relevance')
plt.xlabel('Score')
plt.ylabel('Count')
plt.xticks(range(1, 6))

plt.tight_layout()
plt.savefig('data/llm_judge_score_histograms.png')
plt.show()
