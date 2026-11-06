"""Evaluate retrieval, refusal and answers on evals/cases.json.

    python evals/run.py                       offline: BM25 + extractive answers, no key, free
    python evals/run.py --live MODEL_ID       same retrieval, answers from a model via OpenRouter (needs OPENROUTER_API_KEY)
    python evals/run.py --threshold 0.6       skip tuning and use this refusal threshold

The refusal threshold is chosen on the "dev" cases only. The "test" cases were written with the dev cases, and the author
looked at their scores while debugging, so they are not clean. The "holdout" cases were written before the retrieval
fix and scored afterwards: those are the numbers to trust.

Metrics
- hit@k / MRR: is the chunk holding the answer among the top results? (answerable questions)
- answered: answerable questions the system did not refuse.   refused: unanswerable questions it refused.
- fact: the answer contains every required fact string.   cited: the answer cites the gold chunk.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from ragkit import Index, answer  # noqa: E402

GRID = [round(0.3 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90


def load_cases():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        return json.load(f)


def gold_chunk_ids(index, case):
    return [c.id for c in index.chunks if c.doc.startswith(case["doc"]) and case["gold"] in c.text]


def retrieval_stats(index, cases, k=4):
    rr, hits = [], {1: 0, 3: 0, k: 0}
    ans = [c for c in cases if not c.get("unanswerable")]
    for c in ans:
        gold = set(gold_chunk_ids(index, c))
        assert gold, f"{c['id']}: gold phrase not found in the corpus"
        ranked = [ch.id for ch, _s, _v in index.search(c["q"], k)]
        pos = next((i for i, cid in enumerate(ranked, 1) if cid in gold), None)
        rr.append(1 / pos if pos else 0.0)
        for n in hits:
            hits[n] += bool(pos and pos <= n)
    n = len(ans) or 1
    return {"n": len(ans), "hit1": hits[1] / n, "hit3": hits[3] / n, f"hit{k}": hits[k] / n, "mrr": sum(rr) / n, "k": k}


def run_answers(index, cases, threshold, llm=None):
    rows = []
    for c in cases:
        res = answer(c["q"], index, threshold, llm=llm)
        row = {"id": c["id"], "unanswerable": bool(c.get("unanswerable")), "refused": res["refused"], "reason": res["reason"],
               "answer": res["answer"]}
        if not row["unanswerable"] and not res["refused"]:
            row["fact"] = all(m.lower() in res["answer"].lower() for m in c["must"])
            row["cited"] = bool(set(res["citations"]) & set(gold_chunk_ids(index, c)))
        rows.append(row)
    return rows


def summarize(rows):
    ans = [r for r in rows if not r["unanswerable"]]
    un = [r for r in rows if r["unanswerable"]]
    given = [r for r in ans if not r["refused"]]
    return {
        "answered": sum(not r["refused"] for r in ans), "n_ans": len(ans),
        "refused": sum(r["refused"] for r in un), "n_un": len(un),
        "fact": sum(r["fact"] for r in given), "cited": sum(r["cited"] for r in given), "n_given": len(given),
        "accuracy": (sum(not r["refused"] for r in ans) + sum(r["refused"] for r in un)) / (len(rows) or 1),
    }


def tune_threshold(index, dev):
    scores = {t: summarize(run_answers(index, dev, t))["accuracy"] for t in GRID}
    best = max(scores.values())
    tied = [t for t, a in scores.items() if a == best]
    return tied[len(tied) // 2], scores


def fmt(s):
    return (f"answered {s['answered']}/{s['n_ans']}, refused {s['refused']}/{s['n_un']}, "
            f"fact {s['fact']}/{s['n_given']}, cited {s['cited']}/{s['n_given']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="MODEL")
    ap.add_argument("--threshold", type=float)
    a = ap.parse_args()
    index = Index.from_folder(os.path.join(ROOT, "corpus", "sample"))
    cases = load_cases()
    dev = [c for c in cases if c["split"] == "dev"]
    test = [c for c in cases if c["split"] == "test"]
    hold = [c for c in cases if c["split"] == "holdout"]
    llm = None
    if a.live:
        from ragkit.llm import make_llm
        llm = make_llm(a.live)
    if a.threshold is not None:
        thr, scores = a.threshold, {}
    else:
        thr, scores = tune_threshold(index, dev)
    rd, rt, rh = retrieval_stats(index, dev), retrieval_stats(index, test), retrieval_stats(index, hold)
    sd, st, sh = (summarize(run_answers(index, x, thr, llm)) for x in (dev, test, hold))
    mode = f"live: {a.live}" if a.live else "offline: BM25 + extractive answers"
    lines = [f"# RAG eval ({mode})", "",
             f"- Corpus: {len(index.chunks)} chunks from {len({c.doc for c in index.chunks})} documents. Cases: {len(dev)} dev, {len(test)} test, {len(hold)} holdout.",
             f"- Refusal threshold: **{thr}** " + ("(given)" if a.threshold is not None else "(tuned on dev only)"), "",
             "| Split | hit@1 | hit@3 | hit@4 | MRR |", "|---|---|---|---|---|",
             f"| dev | {rd['hit1']:.0%} | {rd['hit3']:.0%} | {rd['hit4']:.0%} | {rd['mrr']:.2f} |",
             f"| test | {rt['hit1']:.0%} | {rt['hit3']:.0%} | {rt['hit4']:.0%} | {rt['mrr']:.2f} |",
             f"| holdout | {rh['hit1']:.0%} | {rh['hit3']:.0%} | {rh['hit4']:.0%} | {rh['mrr']:.2f} |", "",
             "| Split | Answer and refusal |", "|---|---|", f"| dev | {fmt(sd)} |", f"| test | {fmt(st)} |", f"| holdout | {fmt(sh)} |"]
    if scores:
        lines += ["", "Dev accuracy by threshold: " + ", ".join(f"{t}: {v:.0%}" for t, v in scores.items())]
    text = "\n".join(lines) + "\n"
    print(text)
    if not a.live and a.threshold is None:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "offline-latest.md"), "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
