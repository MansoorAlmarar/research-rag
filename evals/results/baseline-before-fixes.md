# RAG eval (offline: BM25 + extractive answers)

- Corpus: 30 chunks from 5 documents. Cases: 18 dev, 18 test, 12 holdout.
- Refusal threshold: **0.3** (tuned on dev only)

| Split | hit@1 | hit@3 | hit@4 | MRR |
|---|---|---|---|---|
| dev | 92% | 100% | 100% | 0.96 |
| test | 100% | 100% | 100% | 1.00 |
| holdout | 83% | 100% | 100% | 0.92 |

| Split | Answer and refusal |
|---|---|
| dev | answered 11/13, refused 2/5, fact 8/11, cited 11/11 |
| test | answered 11/13, refused 1/5, fact 10/11, cited 11/11 |
| holdout | answered 6/6, refused 3/6, fact 5/6, cited 5/6 |

Dev accuracy by threshold: 0.3: 72%, 0.35: 61%, 0.4: 56%, 0.45: 50%, 0.5: 50%, 0.55: 39%, 0.6: 39%, 0.65: 39%, 0.7: 39%, 0.75: 39%, 0.8: 33%, 0.85: 33%, 0.9: 33%
