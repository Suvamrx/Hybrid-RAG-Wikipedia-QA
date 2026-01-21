# LLM-as-Judge Evaluation: Method and Findings

## Methodology
We implemented an innovative evaluation for our Hybrid RAG system using an LLM-as-Judge approach. For each generated answer, a language model (Flan-T5-base) was prompted to rate the response on three criteria:
- **Factuality**: Is the answer factually correct?
- **Completeness**: Does the answer fully address the question?
- **Relevance**: Is the answer relevant to the question?

Each criterion was scored on a 1–5 scale, and the LLM was also asked to provide a brief explanation for its ratings. The results were saved in a structured JSON file, with nulls recorded where the model did not return a score.

## Results Summary
- **Factuality**: Mean = 3.04, Median = 4.00, Std = 1.44, Nulls = 3
- **Completeness**: Mean = 3.18, Median = 4.00, Std = 1.60, Nulls = 6
- **Relevance**: Mean = 3.50, Median = 4.00, Std = 1.11, Nulls = 2

Most answers received moderate to high scores, with relevance generally rated highest. Some entries had missing scores, likely due to model output limitations.

## Observations
- The LLM-judge approach provides nuanced, human-like evaluation, especially for subjective aspects like completeness and relevance.
- Null values and repeated explanations indicate occasional model limitations or prompt misalignment.
- Score distributions (see attached plots) show a tendency toward higher ratings, but with a meaningful spread across the scale.

## Limitations
- The LLM (Flan-T5-base) sometimes failed to return all requested scores, resulting in nulls.
- Some explanations were repetitive or generic.
- More instruction-tuned models may yield even more reliable results.

## Conclusion
The LLM-as-Judge evaluation complements automated metrics by providing a more holistic, qualitative assessment of answer quality. This method highlights strengths and weaknesses not captured by traditional metrics, supporting a more comprehensive evaluation of RAG system performance.
