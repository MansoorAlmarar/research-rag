# RAG eval (live: claude-cli:haiku)

- Corpus: 30 chunks from 5 documents. Cases: 18 dev, 18 test, 12 holdout.
- Refusal threshold: **0.35** (tuned on dev only)

| Split | hit@1 | hit@3 | hit@4 | MRR |
|---|---|---|---|---|
| dev | 92% | 100% | 100% | 0.96 |
| test | 100% | 100% | 100% | 1.00 |
| holdout | 100% | 100% | 100% | 1.00 |

| Split | Answer and refusal |
|---|---|
| dev | answered 11/13, refused 5/5, fact 11/11, cited 11/11 |
| test | answered 11/13, refused 5/5, fact 11/11, cited 11/11 |
| holdout | answered 6/6, refused 6/6, fact 6/6, cited 6/6 |

Misses:
- `q05`: refused (model said not in documents)
- `q09`: refused (model said not in documents)
- `q18`: refused (low coverage)
- `q20`: refused (low coverage)

Dev accuracy by threshold: 0.3: 89%, 0.35: 94%, 0.4: 83%, 0.45: 72%, 0.5: 72%, 0.55: 78%, 0.6: 61%, 0.65: 56%, 0.7: 50%, 0.75: 50%, 0.8: 44%, 0.85: 44%, 0.9: 44%
