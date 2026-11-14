# RAG eval (live: claude-cli:haiku)

- Corpus: 30 chunks from 5 documents. Cases: 18 dev, 22 test, 12 holdout.
- Refusal threshold: **0.35** (tuned on dev only)
- Every case, with its answer and the reason for each refusal: `evals/results/live-claude-cli-haiku.json`

| Split | hit@1 | hit@3 | hit@4 | MRR |
|---|---|---|---|---|
| dev | 92% | 100% | 100% | 0.96 |
| test | 100% | 100% | 100% | 1.00 |
| holdout | 100% | 100% | 100% | 1.00 |

| Split | Answer and refusal |
|---|---|
| dev | answered 12/13, refused 5/5, fact 12/12, cited 12/12 |
| test | answered 9/13, refused 9/9, fact 9/9, cited 9/9 |
| holdout | answered 6/6, refused 6/6, fact 6/6, cited 6/6 |

Refused by (who refused). An error counts as a refusal above, so a correct refusal is only one made by the gate or the model.

| Split | Unanswerable, refused by | Answerable, refused by |
|---|---|---|
| dev | gate 4, model 1, error 0 | gate 0, model 1, error 0 |
| test | gate 6, model 3, error 0 | gate 2, model 2, error 0 |
| holdout | gate 4, model 2, error 0 | gate 0, model 0, error 0 |

Misses:
- `q05`: refused (model: model said not in documents)
- `q04`: refused (model: model said not in documents)
- `q06`: refused (model: model said not in documents)
- `q18`: refused (gate: low coverage)
- `q20`: refused (gate: low coverage)

Dev accuracy by threshold: 0.3: 100%, 0.35: 100%, 0.4: 89%, 0.45: 78%, 0.5: 78%, 0.55: 78%, 0.6: 61%, 0.65: 56%, 0.7: 50%, 0.75: 50%, 0.8: 44%, 0.85: 44%, 0.9: 44%
