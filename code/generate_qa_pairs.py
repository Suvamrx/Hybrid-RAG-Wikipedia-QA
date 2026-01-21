import json
import os
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# =====================================
# Automated Q&A Pair Generation Script
# =====================================
# This script generates a diverse set of Q&A pairs from Wikipedia chunks using an LLM.
# It samples passages, generates questions and answers, and saves the dataset for evaluation.

# Model and data parameters
MODEL_NAME = 'google/flan-t5-base'
CHUNKS_PATH = os.path.join(os.getcwd(), 'data', 'wikipedia_chunks.json')
QA_PATH = os.path.join(os.getcwd(), 'data', 'generated_qa_pairs.json')
NUM_QA = 100  # Number of Q&A pairs to generate

# Load preprocessed Wikipedia chunks
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Sample a diverse set of chunks for Q&A generation
random.seed(42)
sampled_chunks = random.sample(chunks, min(NUM_QA, len(chunks)))

# Load the language model and tokenizer for Q&A generation
print('Loading LLM for Q&A generation...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print('Using GPU for inference.')
else:
    print('Using CPU for inference.')
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

# Number of retries for non-empty answer/question
max_retries = 3

# Generate a question from a Wikipedia passage using the LLM
# Tries up to max_retries to get a non-empty question


def generate_question(text):
    prompt = (
        "Read the following Wikipedia passage and generate a clear, direct question that could be answered from the passage.\n"
        "Passage: " + text + "\n"
        "Question:"
    )
    for attempt in range(max_retries):
        output = generator(prompt, max_length=64, do_sample=True, top_p=0.95, temperature=0.8)[0]['generated_text']
        print(f"\n[DEBUG] LLM question output (attempt {attempt+1}): {output}\n")
        # Extract question (first non-empty line or after 'Question:')
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        q = ''
        for line in lines:
            if line.lower().startswith('question:'):
                q = line.split(':', 1)[-1].strip()
                break
        if not q and lines:
            q = lines[0]
        if q:
            return q
    return ''

# Generate an answer to a question using the LLM and passage
# Tries up to max_retries to get a non-empty answer


def generate_answer(text, question):
    prompt = (
        f"Read the following Wikipedia passage and answer the question in a detailed, fact-based, and complete manner. "
        f"Your answer should be as informative as possible, using only information from the passage.\n"
        f"Passage: {text}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    for attempt in range(max_retries):
        output = generator(prompt, max_length=256, do_sample=True, top_p=0.95, temperature=0.8)[0]['generated_text']
        print(f"\n[DEBUG] LLM answer output (attempt {attempt+1}): {output}\n")
        # Extract answer (first non-empty line or after 'Answer:')
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        a = ''
        for line in lines:
            if line.lower().startswith('answer:'):
                a = line.split(':', 1)[-1].strip()
                break
        if not a and lines:
            a = lines[0]
        if a:
            return a
    return ''

# Main loop: generate Q&A pairs for each sampled chunk
qa_pairs = []
for i, chunk in enumerate(sampled_chunks):
    print(f"Generating Q&A for chunk {i+1}/{len(sampled_chunks)}: {chunk['title']}")
    q = generate_question(chunk['text'])
    a = generate_answer(chunk['text'], q) if q else ''
    qa_pairs.append({
        'question_id': i,
        'question': q,
        'answer': a,
        'source_url': chunk.get('source_url', chunk.get('url', '')),
        'chunk_id': chunk['chunk_id'],
        'title': chunk['title'],
        'category': '',  # Optionally fill in manually or with heuristics
    })

# Save generated Q&A pairs to file for evaluation
with open(QA_PATH, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

print(f"Saved {len(qa_pairs)} Q&A pairs to {QA_PATH}")
