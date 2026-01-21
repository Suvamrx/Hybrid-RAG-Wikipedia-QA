### Error Analysis Examples

| QID | Question | Ground Truth | Generated Answer | LLM-Judge (F/C/R) | Explanation | Metrics (MRR/F1/ROUGE-L) | Error Type/Comment |
|-----|----------|--------------|------------------|-------------------|-------------|-------------------------|--------------------|
| 2 | Which of the following is not an example of interstellar ion–molecule chemistry: atomic hydrogen or molecular hydrogen? | atomic hydrogen | H2 (hydrogen gas, or dihydrogen) is the most abundant molecule in the universe. | 1/1/4 | Atomic hydrogen is the most abundant molecule in the universe. | 1.0/0.11/0.13 | Factual error: model confuses atomic and molecular hydrogen. |
| 5 | What is the Miami classification? | The Miami classification is the most popular system in oesophageal CLE diagnosis. | Earth. | 1/1/4 | Miami is a city in Florida, United States. | 1.0/0.13/0.0 | Irrelevant/generic answer. |
| 7 | What is the purpose of the Uruz Project? | to recreate the aurochs by a more efficient breeding strategy using genome editing... | Heinz and Lutz Heck attempting to breed the aurochs. | 1/1/4 | The Uruz Project is a project of Heinz and Lutz Heck attempting to breed the aurochs. | 0.25/0.17/0.17 | Incomplete answer: lacks purpose and method. |

*F = Factuality, C = Completeness, R = Relevance*
