# research-rag

Ask questions about a folder of company documents and get a short answer with citations, or an honest "not found". Small on purpose: plain Python, BM25 retrieval, no vector database, no framework. The point is the part most demos skip: **knowing when not to answer, and measuring it.**

![CI](https://github.com/MansoorAlmarar/research-rag/actions/workflows/ci.yml/badge.svg)

```
$ python ask.py "How many vessels does Tidewater run?"
The group operates 214 vessels, of which 151 are owned and 63 are chartered.
Sources: tidewater-logistics-annual#2

$ python ask.py "Who is the chief financial officer of Tidewater Logistics?"
Not found in the documents.
(low coverage)
```

## How it works

1. **Chunk**: each `.md` / `.txt` file is split by `#` / `##` headings, long sections at sentence ends. Every chunk keeps its document title and heading, so "Fleet" is never just "Fleet".
2. **Retrieve**: BM25 over chunks. If the question names a company, only that company's document is searched. The list of names comes from the corpus (the capitalised words of each document title, such as Tidewater or Helio), and a question matches them in any letter case. As a safety net, a question that names no known company but has a capitalised name the corpus never mentions ("France") matches nothing.
3. **Refuse**: a coverage score (how much of the question's weight appears in the best chunk) below a threshold means "Not found in the documents."
4. **Answer**: either extractive (the best sentence, offline, no key) or a model through OpenRouter that must return JSON with citations. The extractive answerer refuses when its best sentence holds under 35% of the question's terms, not counting the company name.
5. **Check the model**: the answer is dropped if it cites no source, or if any cited id is not a retrieved chunk id. It is also dropped if it states a number that is not in the cited text (digits only: names and spelled-out numbers are not checked). Bad JSON, an empty answer or an API error become a refusal, never a crash.

Every refusal records who refused: the coverage gate, the model, the extractive answerer, or an error. The eval reports that split, so a failed call is never mistaken for a correct "not found".

## Evaluation

The sample corpus is 5 fictional company documents (30 chunks). `evals/cases.json` has 52 labelled questions: 32 answerable (paraphrased, not copied from the text) and 20 unanswerable, 10 of which are deliberate near-misses, such as asking for Tidewater's CFO when only another company's CFO is mentioned.

Cases are split three ways, and the split matters:
- **dev** (18): used to choose the refusal threshold (0.35) and the extractive overlap floor (0.35).
- **test** (22): written with dev, but I looked at its scores while debugging, so it is **not clean**. It includes 4 questions (`n01` to `n04`) added in the second round of fixes: lowercase and reworded versions of earlier unanswerable questions. They were written after the fix they test, so they show that the fix works, not that it generalises.
- **holdout** (12): written *before* the first retrieval fix and scored after. The history here is only two commits and the first already holds those fixes, so this order rests on my word, not on the git log. The second round of fixes was designed after reading the holdout failures, so the holdout is **no longer clean** either.

Offline run (`python evals/run.py`), BM25 + extractive answers:

| Split | hit@1 | hit@3 | MRR | Answered | Refused | Right fact | Cited gold chunk |
|---|---|---|---|---|---|---|---|
| dev | 92% | 100% | 0.96 | 13/13 | 5/5 | 11/13 | 13/13 |
| test | 100% | 100% | 1.00 | 11/13 | 8/9 | 11/11 | 11/11 |
| holdout | 100% | 100% | 1.00 | 6/6 | 5/6 | 6/6 | 6/6 |

("Answered" is over answerable questions, "Right fact" is over the questions that were answered, and "Refused" is over unanswerable ones.)

Who refused the 18 unanswerable questions that were refused: the coverage gate 14, the extractive answerer 4. The two that got an answer are `h09` (Orbit's *operating* margin, answered with the gross margin) and `n04` ("what is the capital of france?" in lowercase). Each run also writes `evals/results/offline-latest.json` with one row per question: the answer, whether it was refused, by whom and why. Every number above can be checked against it.

**What changed it.** The first holdout run, before any fix, scored hit@1 83%, refused 3/6, right fact 5/6 (`evals/results/baseline-before-fixes.md`). Three fixes, each found by reading failures rather than guessing:
- `'s` was tokenised into a stray `s`, and question words like "much" and "many" counted against coverage.
- A question naming one company still matched another company's chunk on shared words ("chief financial officer"). The entity filter fixed that.
- The extractive answerer sometimes returned the chunk's title line as the "answer".

**Second round.** A code review found four more problems. Before these fixes the offline run refused 11 of the 16 unanswerable questions it had then. After, it refuses 15 of those same 16, and answers the same 30 of 32 answerable ones.
- The company filter only saw capital letters. "who is the cfo of tidewater logistics?" in lowercase returned another company's CFO. The names now come from the document titles and match in any case.
- Four unanswerable questions got the document's intro line back ("Helio Solar is a fictional manufacturer..."), because it repeats the company name. The extractive answerer now needs a sentence that holds at least 35% of the question's other terms. That floor was chosen on dev only, and the margin is thin: the weakest correct dev answer sits at 40%.
- "Right fact" was a plain substring check, so "9%" could be found inside "29%". It now matches whole numbers and words. No score changed, but the check can now be trusted.
- One made-up citation next to a real one used to pass. Now any invalid citation drops the answer.

## With a model

Same retrieval and the same refusal gate, but a model writes the answer from the retrieved chunks and must cite them (`python evals/run.py --live claude-cli:haiku`, results in `evals/results/live-claude-cli-haiku.md`, one row per question in `live-claude-cli-haiku.json`). Claude Haiku through the local `claude` CLI, scored on the same 52 questions, run on 2026-11-14 after the second round:

| Split | Answered | Refused | Right fact | Cited gold chunk |
|---|---|---|---|---|
| dev | 12/13 | 5/5 | 12/12 | 12/12 |
| test | 9/13 | 9/9 | 9/9 | 9/9 |
| holdout | 6/6 | 6/6 | 6/6 | 6/6 |

Against the offline version, the model fixes the weak spot: all 20 unanswerable questions were refused. The per-question file shows who refused them: the coverage gate 14, the model itself 6 ("not in the documents"), errors 0. So none of the 20 is a failed call counted as a refusal. The 6 the model refused include the two the offline version gets wrong (`h09`, the operating margin, and `n04`, the lowercase France question). Every answer it gave had the right fact and cited the right chunk (27 of 27).

What it still misses is on the answerable side, 5 of 32:
- `q18` and `q20` are refused by the lexical gate before the model sees them, because "operate" and "online" do not appear in the text ("640 stores", "e-commerce was 27%").
- `q04`, `q05` and `q06` reach the model with the right chunk, and it says "not in the documents". This is the model being too careful, and it varies between runs: the earlier run (48 questions, older code) refused `q05` and `q09` and answered `q04` and `q06`. One run is saved here, so read the answered count as plus or minus two questions.

How it was run: the CLI is called with all tools off, from an empty temporary folder (no project instructions), with the prompt on stdin. It uses a Claude subscription's plan usage and no API key. OpenRouter models also work (`--live provider/model`, needs a key; `:free` models work with $0 credit).

## What it does not do well (honest limits)

- **One holdout near-miss still gets answered.** Asking for Orbit's *operating* margin returns "Gross margin was 64%." A word-overlap score cannot tell "margin" from "operating margin". Embeddings or a reranker would help, and so does letting a model decide (`--model`), which can say `NOT_IN_DOCUMENTS`. That is what the "With a model" run above shows.
- **An unknown name in lowercase slips through.** "What is the capital of France?" is refused, because France is a capitalised name the corpus never mentions. "what is the capital of france?" is answered with a sentence about capital expenditure. The company filter no longer depends on capitals, but this safety net still does. It is in the eval as `n04`.
- **Company names come from document titles.** A document with no `#` title, or a company that is only mentioned inside another company's document, is not a name the filter knows.
- **An extra unknown word can cost an answer.** "How many vessels does Tidewater run, according to Bloomberg?" finds the right chunk but is refused, because the unknown words pull coverage under the threshold.
- **Lexical only.** "Fraction" will not find "share", "online" will not find "e-commerce". Retrieval still scores 100% hit@3 here because the corpus is small and clean.
- **Small, fake corpus and one labeller.** 52 questions on 5 invented documents. One question is about 2 points. Real filings are longer, messier and contain tables, which this chunker does not parse.
- **Two settings tuned on 18 dev questions.** The refusal threshold and the overlap floor will not transfer to a different corpus without re-tuning.
- **No clean split is left.** Test and holdout have both been read while fixing. The next honest step is a fresh set of questions written before any more changes.
- **Extractive answers are a baseline.** One sentence, no reasoning, no combining facts across chunks.

## Run it

```
pip install -r requirements.txt
python ask.py "What was Helio Solar's gross margin guidance?"        # offline
python ask.py --docs path/to/your/notes "your question"               # your own .md / .txt files
python evals/run.py                                                   # offline eval, writes evals/results/
python ask.py --model claude-cli:haiku "question"                            # Claude through the local CLI, no API key
export OPENROUTER_API_KEY=...                                         # then, for model answers:
python ask.py --model nvidia/nemotron-3-ultra-550b-a55b:free "question"
python evals/run.py --live nvidia/nemotron-3-ultra-550b-a55b:free
python -m pytest -q                                                   # 77 tests, offline, includes eval floors
```

CI runs the tests (which include a regression floor on the eval) on every push and pull request. Keys are read from the environment and never printed. All sample documents are fictional.

## Layout

| Path | What |
|---|---|
| `ragkit/text.py` | tokens, stemming, title names, chunking |
| `ragkit/index.py` | BM25 index, coverage score, company names from the corpus, company filter |
| `ragkit/answer.py` | refusal, extractive answerer, model-answer checks |
| `ragkit/llm.py` | OpenRouter call, Claude CLI call |
| `corpus/sample/` | 5 fictional company documents |
| `evals/` | labelled cases, runner, results (a summary and a per-question JSON file for each run) |
| `tests/` | 77 offline tests |
