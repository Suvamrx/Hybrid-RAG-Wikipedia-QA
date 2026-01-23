"""
Script: run_pipeline.py
Description: Orchestrates the full Hybrid RAG workflow by executing each required script in sequence.
Usage: python code/run_pipeline.py [--skip-step ...] [--only-step ...]
"""
import subprocess
import sys
import os
import argparse

# =====================================
# Hybrid RAG Pipeline Orchestration Script
# =====================================
# This script automates the execution of all major steps in the Hybrid RAG workflow.
# It runs each required script in order, supports skipping or running specific steps,
# and logs progress/errors for reproducibility and ease of use.

# List of pipeline steps and their corresponding scripts
# Each tuple: (Description, Script/Command)
PIPELINE_STEPS = [
    ("Collect fixed Wikipedia URLs", "collect_fixed_wikipedia_urls.py"),
    ("Sample random Wikipedia URLs", "sample_random_wikipedia_urls.py"),
    ("Preprocess and chunk Wikipedia articles (fixed)", "preprocess_and_chunk_wikipedia.py --use-fixed"),
    ("Preprocess and chunk Wikipedia articles (random)", "preprocess_and_chunk_wikipedia.py --use-random"),
    ("Build dense vector index (FAISS)", "dense_retrieval_faiss.py"),
    ("Build sparse index (BM25)", "sparse_retrieval_bm25.py"),
    ("Reciprocal Rank Fusion (RRF)", "reciprocal_rank_fusion.py"),
    ("Generate Q&A pairs", "generate_qa_pairs.py"),
    ("Run full RAG evaluation pipeline", "evaluate_rag_pipeline.py"),
    ("Run dense-only ablation", "evaluate_rag_pipeline_dense.py"),
    ("Run sparse-only ablation", "evaluate_rag_pipeline_sparse.py"),
    ("LLM-as-Judge Evaluation", "llm_judge_evaluation.py"),
    ("Generate report (HTML/PDF)", "generate_report.py"),
    ("Generate evaluation visualizations", "evaluation_visualizations.py"),
]

# Parse command-line arguments for skipping or running specific steps
parser = argparse.ArgumentParser(description="Run the full Hybrid RAG pipeline.")
parser.add_argument('--skip-step', nargs='*', default=[], help='Step numbers to skip (e.g., --skip-step 2 3)')
parser.add_argument('--only-step', nargs='*', default=[], help='Only run these step numbers (e.g., --only-step 1 5)')
args = parser.parse_args()

cwd = os.path.join(os.getcwd(), 'code')

# Helper function to run a single pipeline step
# Prints progress, runs the command, and checks for errors

def run_step(idx, desc, script):
    print(f"\n[STEP {idx+1}] {desc}")
    # For non-interactive run, set environment variable to signal scripts to skip input()
    env = os.environ.copy()
    env["NON_INTERACTIVE"] = "1"
    cmd = f"python {script}" if script.endswith('.py') else f"python {script}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Step {idx+1} failed: {desc}")
        sys.exit(result.returncode)
    print(f"[DONE] Step {idx+1}: {desc}")

# Main orchestration loop: run each step in order, respecting skip/only flags
if __name__ == '__main__':
    for idx, (desc, script) in enumerate(PIPELINE_STEPS):
        step_num = str(idx+1)
        if args.only_step and step_num not in args.only_step:
            continue
        if step_num in args.skip_step:
            print(f"[SKIP] Step {step_num}: {desc}")
            continue
        run_step(idx, desc, script)
    print("\n[INFO] Pipeline completed successfully.")
