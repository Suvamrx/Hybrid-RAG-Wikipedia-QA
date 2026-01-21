# Hybrid RAG System Assignment

## Project Structure
- data/: Wikipedia URLs, processed corpus, vector DB, question dataset, evaluation results
- code/: Scripts and notebooks for RAG system, data processing, retrieval, generation
- evaluation/: Question generation, metrics, evaluation pipeline
- reports/: PDF/HTML reports, visualizations, screenshots

## Step-by-Step Instructions

### 1. Environment Setup
```
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```


### 2. Data Collection
Generate 200 fixed Wikipedia URLs:
```
python code/collect_fixed_wikipedia_urls.py
```

Sample N random Wikipedia URLs (overnight, optional):
```
python code/sample_random_wikipedia_urls.py
```

### 3. Preprocessing and Chunking
Download, clean, and chunk Wikipedia articles. You can use fixed URLs, random URLs, or both:
```
# Use only fixed URLs:
python code/preprocess_and_chunk_wikipedia.py --use-fixed
# Use only random URLs:
python code/preprocess_and_chunk_wikipedia.py --use-random
# Use both fixed and random URLs:
python code/preprocess_and_chunk_wikipedia.py --use-fixed --use-random
```
Download NLTK punkt tokenizer (run once):
```
python
>>> import nltk
>>> nltk.download('punkt')
exit()
```

### 4. Dense Retrieval (FAISS)
Build dense vector index and test retrieval:
```
python code/dense_retrieval_faiss.py
```

### 5. Sparse Retrieval (BM25)
Build BM25 index and test retrieval:
```
python code/sparse_retrieval_bm25.py
```

### 6. Reciprocal Rank Fusion (RRF)
Combine dense and sparse results:
```
python code/reciprocal_rank_fusion.py
```

### 7. Response Generation (LLM)
Generate answers using RAG pipeline:
```
python code/generate_response_llm.py
```

### 8. User Interface
Launch Streamlit app for interactive QA:
```
streamlit run code/app_streamlit.py
```

### 9. Automated Q&A Generation
Generate 100 diverse Q&A pairs:
```
python code/generate_qa_pairs.py
```

*Note: The script automatically uses GPU if available, otherwise falls back to CPU. Q&A generation uses a two-step process for better answer quality.*

### 10. Evaluation Pipeline
Run evaluation and compute metrics (MRR, F1, ROUGE-L):
```
python code/evaluate_rag_pipeline.py
```

#### Example Results (latest run):
```
Mean Reciprocal Rank (MRR): 0.8887
Mean F1 Score: 0.0925
Mean ROUGE-L: 0.1036
Results saved to data/evaluation_results.json
```

### 11. Report Generation
Generate a PDF/HTML report from the evaluation results and report template:
```
python code/generate_report.py
```
- The script will create reports/final_report.html and reports/final_report.pdf.
- You can edit reports/final_report_template.md to update the report content or add new results/visualizations.
- For PDF export, you must install [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) and ensure it is in your system PATH. If you only need HTML, this is not required.

### 12. Ablation Studies: Dense-Only and Sparse-Only Evaluation
To reproduce ablation results for dense-only and sparse-only retrieval (as reported in the final report):

#### Dense-Only Evaluation
```
python code/evaluate_rag_pipeline_dense.py
```
- Computes metrics using only the dense retriever (FAISS).
- Results saved to data/evaluation_results_dense.json.

#### Sparse-Only Evaluation
```
python code/evaluate_rag_pipeline_sparse.py
```
- Computes metrics using only the sparse retriever (BM25).
- Results saved to data/evaluation_results_sparse.json.

### 13. Full Pipeline Orchestration
To run the entire Hybrid RAG workflow automatically, use the orchestration script:
```
python code/run_pipeline.py
```
- Runs all required steps in sequence: data collection, chunking, retrieval, evaluation, ablation, and report generation.
- Use `--skip-step` or `--only-step` flags to skip or run specific steps (e.g., `--skip-step 2 3` or `--only-step 1 5`).
- Logs progress and errors for reproducibility.

#### Example Usage for Full Pipeline Script

Run all steps in sequence:
```
python code/run_pipeline.py
```

Skip random URL sampling and ablation steps:
```
python code/run_pipeline.py --skip-step 2 9 10
```

Run only data collection and chunking:
```
python code/run_pipeline.py --only-step 1 2 3 4
```

Run only evaluation and report generation:
```
python code/run_pipeline.py --only-step 8 11
```

Refer to the script's help for more options:
```
python code/run_pipeline.py --help
```

This script is recommended for reproducible experiments and easy setup.

#### Pipeline Step Mapping

The pipeline steps correspond to the following operations:

1. Data Collection (fixed URLs)
2. Data Collection (random URLs)
3. Wikipedia Chunking
4. Chunk Metadata Generation
5. Dense Retrieval (FAISS)
6. Sparse Retrieval (BM25)
7. Reciprocal Rank Fusion (RRF)
8. Q&A Pair Generation
9. Evaluation (Hybrid)
10. Evaluation (Dense/Sparse)
11. Report Generation

Use these step numbers with `--skip-step` or `--only-step` flags in the orchestration script.

## Submission Checklist
- data/fixed_urls.json (200 URLs)
- data/wikipedia_chunks.json (processed corpus)
- data/faiss_index.bin, data/chunk_metadata.json (vector DB)
- data/generated_qa_pairs.json (100-question dataset)
- data/evaluation_results.json (evaluation results)
- code/: All scripts
- README.md: Updated instructions
- Streamlit app link or setup instructions

## Notes
- For best results, ensure prompts are clear and context-grounded.
- If answer quality metrics are low, review Q&A pairs and answer generation prompts.

### Limitations & Future Improvements
- Some generated answers may be generic or not directly address the question, especially for definition-type ("What is...") queries. This is due to the content of retrieved chunks and model limitations.
- Improving chunking strategy, retrieval, or using a more instruction-tuned model could further enhance answer quality.
- For best results, ensure that Wikipedia chunks include clear definitions and introductory sentences for key terms.
- These limitations are acknowledged as part of the current system and are common in RAG pipelines.

## Tips & Troubleshooting
- For faster Q&A generation, use a machine with an NVIDIA GPU and the correct CUDA drivers.
- If you run out of memory, reduce the number of Q&A pairs or use a smaller model (see code/generate_qa_pairs.py).
- If answers are too short or generic, further tune the answer prompt in generate_qa_pairs.py.
