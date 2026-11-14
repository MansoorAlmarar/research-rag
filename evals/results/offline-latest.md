# RAG eval (offline: BM25 + extractive answers)

- Corpus: 30 chunks from 5 documents. Cases: 18 dev, 22 test, 12 holdout.
- Refusal threshold: **0.35** (tuned on dev only)
- Every case, with its answer and the reason for each refusal: `evals/results/offline-latest.json`

| Split | hit@1 | hit@3 | hit@4 | MRR |
|---|---|---|---|---|
| dev | 92% | 100% | 100% | 0.96 |
| test | 100% | 100% | 100% | 1.00 |
| holdout | 100% | 100% | 100% | 1.00 |

| Split | Answer and refusal |
|---|---|
| dev | answered 13/13, refused 5/5, fact 11/13, cited 13/13 |
| test | answered 11/13, refused 8/9, fact 11/11, cited 11/11 |
| holdout | answered 6/6, refused 5/6, fact 6/6, cited 6/6 |

Refused by (who refused). An error counts as a refusal above, so a correct refusal is only one made by the gate or the extractive answerer.

| Split | Unanswerable, refused by | Answerable, refused by |
|---|---|---|
| dev | gate 4, extractive 1 | gate 0, extractive 0 |
| test | gate 6, extractive 2 | gate 2, extractive 0 |
| holdout | gate 4, extractive 1 | gate 0, extractive 0 |

Misses:
- `q13`: wrong fact
- `q15`: wrong fact
- `q18`: refused (gate: low coverage)
- `q20`: refused (gate: low coverage)
- `n04`: answered an unanswerable question
- `h09`: answered an unanswerable question

Dev accuracy by threshold: 0.3: 100%, 0.35: 100%, 0.4: 89%, 0.45: 78%, 0.5: 78%, 0.55: 78%, 0.6: 61%, 0.65: 56%, 0.7: 50%, 0.75: 50%, 0.8: 44%, 0.85: 44%, 0.9: 44%
