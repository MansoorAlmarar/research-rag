# research-rag

Ask questions about a folder of company documents and get a short answer with citations, or an honest "not found". Small on purpose: plain Python, BM25 retrieval, no vector database, no framework. The point is the part most demos skip: **knowing when not to answer, and measuring it.**

![CI](https://github.com/MansoorAlmarar/research-rag/actions/workflows/ci.yml/badge.svg)

```
$ python ask.py "How many vessels does Tidewater run?"
The group operates 214 vessels, of which 151 are owned and 63 are chartered.
Sources: tidewater-logistics-annual#2

$ python ask.py "Who is the chief financial officer of Tidewater Logistics?"
Not found in the documents.
```

## How it works

1. **Chunk**: each `.md` / `.txt` file is split by `#` / `##` headings, long sections at sentence ends. Every chunk keeps its document title and heading, so "Fleet" is never just "Fleet".
2. **Retrieve**: BM25 over chunks. If the question names a company (a capitalised word), only that company's chunks are searched. A name that appears nowhere matches nothing.
3. **Refuse**: a coverage score (how much of the question's weight appears in the best chunk) below a threshold means "Not found in the documents."
4. **Answer**: either extractive (the best sentence, offline, no key) or a model through OpenRouter that must return JSON with citations.
5. **Check the model**: its answer is dropped if a citation is not a retrieved chunk id, or if it states a number that is not in the cited text. Bad JSON, an empty answer or an API error become a refusal, never a crash.

## Evaluation

The sample corpus is 5 fictional company documents (30 chunks). `evals/cases.json` has 40 labelled questions: 32 answerable (paraphrased, not copied from the text) and 12 unanswerable, 6 of which are deliberate near-misses, such as asking for Tidewater's CFO when only another company's CFO is mentioned.

Cases are split three ways, and the split matters:
- **dev**: used to choose the refusal threshold (0.35).
- **test**: written with dev, but I looked at its scores while debugging, so it is **not clean**.
- **holdout**: 12 questions written *before* the retrieval fix and scored after. Trust these.

Offline run (`python evals/run.py`), BM25 + extractive answers:

| Split | hit@1 | hit@3 | MRR | Answered | Refused | Right fact | Cited gold chunk |
|---|---|---|---|---|---|---|---|
| dev | 92% | 100% | 0.96 | 13/13 | 4/5 | 11/13 | 13/13 |
| test | 100% | 100% | 1.00 | 11/13 | 3/5 | 10/11 | 11/11 |
| **holdout** | 100% | 100% | 1.00 | 6/6 | 4/6 | 6/6 | 6/6 |

("Answered" and "Right fact" are over answerable questions; "Refused" is over unanswerable ones.)

**What changed it.** The first holdout run, before any fix, scored hit@1 83%, refused 3/6, right fact 5/6 (`evals/results/baseline-before-fixes.md`). Three fixes, each found by reading failures rather than guessing:
- `'s` was tokenised into a stray `s`, and question words like "much" and "many" counted against coverage.
- A question naming one company still matched another company's chunk on shared words ("chief financial officer"). The entity filter fixed that.
- The extractive answerer sometimes returned the chunk's title line as the "answer".

## What it does not do well (honest limits)

- **Two holdout near-misses still get answered.** Asking for Orbit's *operating* margin, or the *interest rate* on Tidewater's debt, matches chunks that mention margin or interest. A word-overlap score cannot tell "margin" from "operating margin". Embeddings or a reranker would help, and so does letting a model decide (`--model`), which can say `NOT_IN_DOCUMENTS`. I have not scored the live model mode yet, so I make no claim about it.
- **Lexical only.** "Fraction" will not find "share", "online" will not find "e-commerce". Retrieval still scores 100% hit@3 here because the corpus is small and clean.
- **Small, fake corpus and one labeller.** 40 questions on 5 invented documents. One question is 2.5 points. Real filings are longer, messier and contain tables, which this chunker does not parse.
- **Threshold tuned on 18 dev questions.** It will not transfer to a different corpus without re-tuning.
- **Extractive answers are a baseline.** One sentence, no reasoning, no combining facts across chunks.

## Run it

```
pip install -r requirements.txt
python ask.py "What was Helio Solar's gross margin guidance?"        # offline
python ask.py --docs path/to/your/notes "your question"               # your own .md / .txt files
python evals/run.py                                                   # offline eval, writes evals/results/
export OPENROUTER_API_KEY=...                                         # then, for model answers:
python ask.py --model nvidia/nemotron-3-ultra-550b-a55b:free "question"
python evals/run.py --live nvidia/nemotron-3-ultra-550b-a55b:free
python -m pytest -q                                                   # 24 tests, offline, includes eval floors
```

CI runs the tests (which include a regression floor on the eval) on every push and pull request. Keys are read from the environment and never printed. All sample documents are fictional.

## Layout

| Path | What |
|---|---|
| `ragkit/text.py` | tokens, stemming, entity detection, chunking |
| `ragkit/index.py` | BM25 index, coverage score, entity filter |
| `ragkit/answer.py` | refusal, extractive answerer, model-answer checks |
| `ragkit/llm.py` | OpenRouter call |
| `corpus/sample/` | 5 fictional company documents |
| `evals/` | labelled cases, runner, results |
| `tests/` | 24 offline tests |
