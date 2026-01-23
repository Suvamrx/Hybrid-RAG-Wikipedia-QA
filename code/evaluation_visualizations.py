"""
Script: evaluation_visualizations.py
Description: Generates visualizations for evaluation metrics and error analysis for the Hybrid RAG system.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_JSON = os.path.join(PROJECT_ROOT, 'data', 'evaluation_results.json')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load evaluation results
with open(EVAL_JSON, 'r', encoding='utf-8') as f:
    results = json.load(f)
df = pd.DataFrame(results)

# Histograms for each metric
for metric in ['mrr', 'f1', 'rougeL']:
    plt.figure(figsize=(6,4))
    sns.histplot(df[metric], bins=20, kde=True)
    plt.title(f'{metric.upper()} Distribution')
    plt.xlabel(metric.upper())
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'{metric}_hist.png'))
    plt.close()

# Scatter plot: F1 vs. MRR
plt.figure(figsize=(6,6))
sns.scatterplot(x='mrr', y='f1', data=df)
plt.title('F1 vs. MRR')
plt.xlabel('MRR')
plt.ylabel('F1')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'f1_vs_mrr_scatter.png'))
plt.close()

# Error analysis: show top failure types if available
if 'error_type' in df.columns:
    plt.figure(figsize=(8,4))
    df['error_type'].value_counts().plot(kind='bar')
    plt.title('Error Type Distribution')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'error_type_bar.png'))
    plt.close()

print("[INFO] Visualizations saved to reports/ directory.")
