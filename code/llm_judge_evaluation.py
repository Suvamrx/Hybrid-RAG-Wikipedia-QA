
import json
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Parameters
MODEL_NAME = 'google/flan-t5-base'  # Or any open-source LLM you prefer
EVAL_RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'evaluation_results.json')
JUDGE_RESULTS_PATH = os.path.join(os.getcwd(), 'data', 'llm_judge_results.json')


# Load evaluation results (with generated answers)
with open(EVAL_RESULTS_PATH, 'r', encoding='utf-8') as f:
    eval_results = json.load(f)

# Load model and tokenizer
print('Loading LLM for judging...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

def build_single_score_prompt(question, context, answer, criterion, description):
    return (
        f"You are an expert evaluator. Given the following context, question, and answer, rate the answer for {criterion} (1-5):\n"
        f"{description}\n"
        f"Reply with only a single integer (1-5).\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n"
    )

def build_explanation_prompt(question, context, answer):
    return (
        f"You are an expert evaluator. Given the following context, question, and answer, provide a brief explanation (1-2 sentences) of the answer's quality.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\nExplanation:"
    )

criteria = [
    ("factuality", "Is the answer factually correct based only on the provided context? (1=incorrect, 5=fully correct)"),
    ("completeness", "Does the answer fully address the question using the context? (1=incomplete, 5=fully complete)"),
    ("relevance", "Is the answer relevant and grounded in the context? (1=irrelevant, 5=fully relevant)")
]

judge_results = []
for i, r in enumerate(eval_results):
    print(f"Judging Q{i+1}/{len(eval_results)}...")
    question = r['question']
    context = r.get('context', '')
    answer = r['generated_answer']
    scores = {}
    for crit, desc in criteria:
        prompt = build_single_score_prompt(question, context, answer, crit, desc)
        output = generator(prompt, max_length=8, do_sample=False)[0]['generated_text']
        print(f"[RAW LLM OUTPUT Q{i+1} {crit}]: {output}")
        # Extract first integer 1-5
        m = re.search(r"([1-5])", output)
        scores[crit] = int(m.group(1)) if m else None
    # Explanation
    exp_prompt = build_explanation_prompt(question, context, answer)
    exp_output = generator(exp_prompt, max_length=64, do_sample=False)[0]['generated_text']
    print(f"[RAW LLM OUTPUT Q{i+1} explanation]: {exp_output}")
    judge_results.append({
        'question_id': r['question_id'],
        'factuality': scores['factuality'],
        'completeness': scores['completeness'],
        'relevance': scores['relevance'],
        'explanation': exp_output.strip()
    })

with open(JUDGE_RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(judge_results, f, indent=2, ensure_ascii=False)
print(f"LLM judge results saved to {JUDGE_RESULTS_PATH}")
