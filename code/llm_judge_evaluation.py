import json
import os
import re
import csv
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Parameters
MODEL_NAME = 'google/flan-t5-base'  # Or any open-source LLM you prefer
EVAL_RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'evaluation_results.json')
JUDGE_RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'llm_judge_results.json')
BATCH_SIZE = 10  # Number of questions to process per batch
RESUME = True   # Resume from last saved result if True

# Load previous results if resuming
if RESUME and os.path.exists(JUDGE_RESULTS_PATH):
    with open(JUDGE_RESULTS_PATH, 'r', encoding='utf-8') as f:
        judge_results = json.load(f)
    done_ids = {r['question_id'] for r in judge_results}
    start_idx = len(judge_results)
    print(f"[INFO] Resuming from question {start_idx+1} (already completed: {len(judge_results)})")
else:
    judge_results = []
    done_ids = set()
    start_idx = 0

try:
    for i, r in enumerate(eval_results):
        if r['question_id'] in done_ids:
            continue
        print(f"Judging Q{i+1}/{len(eval_results)}...")
        question = r['question']
        context = r.get('context', '')
        answer = r['generated_answer']
        scores = {}
        for crit, desc in criteria:
            prompt = build_single_score_prompt(question, context, answer, crit, desc)
            try:
                output = generator(prompt, max_length=8, do_sample=False)[0]['generated_text']
                print(f"[RAW LLM OUTPUT Q{i+1} {crit}]: {output}")
                m = re.search(r"([1-5])", output)
                scores[crit] = int(m.group(1)) if m else None
            except Exception as e:
                print(f"[ERROR] LLM failed for {crit} on Q{i+1}: {e}")
                scores[crit] = None
        # Explanation
        try:
            exp_prompt = build_explanation_prompt(question, context, answer)
            exp_output = generator(exp_prompt, max_length=64, do_sample=False)[0]['generated_text']
            print(f"[RAW LLM OUTPUT Q{i+1} explanation]: {exp_output}")
        except Exception as e:
            print(f"[ERROR] LLM failed for explanation on Q{i+1}: {e}")
            exp_output = ''
        judge_results.append({
            'question_id': r['question_id'],
            'factuality': scores['factuality'],
            'completeness': scores['completeness'],
            'relevance': scores['relevance'],
            'explanation': exp_output.strip()
        })
        # Save batch results
        if (len(judge_results) % BATCH_SIZE == 0) or (i == len(eval_results)-1):
            with open(JUDGE_RESULTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(judge_results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved {len(judge_results)} results to {JUDGE_RESULTS_PATH}")
except KeyboardInterrupt:
    print("[INFO] Interrupted. Saving progress...")
    with open(JUDGE_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(judge_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(judge_results)} results to {JUDGE_RESULTS_PATH}")

# Save as CSV
CSV_PATH = os.path.join(os.getcwd(), 'data', 'llm_judge_results.csv')
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['question_id', 'factuality', 'completeness', 'relevance', 'explanation'])
    writer.writeheader()
    for row in judge_results:
        writer.writerow(row)
print(f"[INFO] LLM judge results also saved to {CSV_PATH}")

# Visualization (histograms)
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(judge_results)
    for crit in ['factuality', 'completeness', 'relevance']:
        plt.figure(figsize=(6,4))
        df[crit].dropna().astype(int).hist(bins=[1,2,3,4,5,6], rwidth=0.8)
        plt.title(f'{crit.capitalize()} Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'reports', f'llm_judge_{crit}_hist.png'))
        plt.close()
    print("[INFO] LLM judge score histograms saved to reports/ directory.")
except Exception as e:
    print(f"[WARN] Could not generate LLM judge visualizations: {e}")
