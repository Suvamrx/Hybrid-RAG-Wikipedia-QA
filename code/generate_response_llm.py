import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from reciprocal_rank_fusion import reciprocal_rank_fusion, retrieve_dense, retrieve_sparse

# =====================================
# LLM-based Response Generation Script
# =====================================
# This script uses a seq2seq language model (e.g., Flan-T5) to generate answers to user queries
# using context retrieved from the hybrid RAG pipeline (dense, sparse, and RRF fusion).
# It provides functions for context building and answer generation, and supports interactive use.

# Model and retrieval parameters
MODEL_NAME = 'google/flan-t5-base'  # You can change to another open-source model if needed
MAX_INPUT_TOKENS = 1024
TOP_N = 5

# Load the language model and tokenizer for answer generation
print('Loading LLM...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Build context for a query by retrieving top-N chunks using RRF fusion
# Returns the concatenated context string and the fused chunk metadata

def build_context(query, top_n=TOP_N):
    dense_results = retrieve_dense(query, top_k=20)
    sparse_results = retrieve_sparse(query, top_k=20)
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=top_n)
    context = "\n\n".join([f"Source: {r['title']}\n{r['text']}" for r in fused])
    return context, fused

# Generate an answer to a query using the provided context
# Uses the LLM to produce a concise, factual answer based only on the context

def generate_answer(query, context):
    prompt = (
        f"You are a helpful assistant. Answer the following question using ONLY the provided context. "
        f"Be concise, factual, and do not add information not present in the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    # Truncate prompt if too long for the model
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS)
    output = generator(tokenizer.decode(inputs['input_ids'][0]), max_length=128, do_sample=False)[0]['generated_text']
    return output

if __name__ == '__main__':
    # Example usage: interactive query
    query = input('Enter your query: ')
    context, fused = build_context(query)
    print('\n--- Context Used ---')
    print(context)
    print('\n--- Generating Answer ---')
    answer = generate_answer(query, context)
    print(f'\nAnswer:\n{answer}')
