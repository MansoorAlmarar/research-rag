"""Evaluate retrieval, refusal and answers on evals/cases.json.

    python evals/run.py                       offline: BM25 + extractive answers, no key, free
    python evals/run.py --live MODEL_ID       same retrieval, answers from a model (OpenRouter id, or claude-cli:haiku)
    python evals/run.py --threshold 0.6       skip tuning and use this refusal threshold (prints only, writes no files)

The refusal threshold is chosen on the "dev" cases only. The "test" cases were written with the dev cases, and the author
looked at their scores while debugging, so they are not clean. The "holdout" cases were written before the retrieval
fix and scored afterwards.

Metrics
- hit@k / MRR: is the chunk holding the answer among the top results? (answerable questions)
- answered: answerable questions the system did not refuse.   refused: unanswerable questions it refused.
- fact: the answer contains every required fact string, as a whole number or word.   cited: the answer cites the gold chunk.
- refused by: who refused. gate = coverage below the threshold. model = the model said NOT_IN_DOCUMENTS.
  extractive = the offline answerer found no sentence. error = the call failed or the reply failed a check.
  An error on an unanswerable question still counts as "refused", so read this split before trusting that number.

Each run writes two files to evals/results/: a Markdown summary and a JSON file with one row per case
(id, refused, refused_by, reason, answer and more), so every number in the summary can be checked.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from ragkit import Index, answer  # noqa: E402

GRID = [round(0.3 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
REFUSED_BY = ("gate", "model", "extractive", "error")


def load_cases():
    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        return json.load(f)


def gold_chunk_ids(index, case):
    return [c.id for c in index.chunks if c.doc.startswith(case["doc"]) and case["gold"] in c.text]


def has_fact(must, text):
    """True when `must` is in `text` as a whole number or word, ignoring case.
    "9%" is not found inside "29%", "310" is not found inside "KB-310", and "0.9" is not found inside "0.95".
    A unit letter after a number is fine: "412" is found in "$412M"."""
    pattern = r"(?<![0-9A-Za-z.,-])" + re.escape(must) + r"(?![0-9])(?![.,][0-9])"
    return re.search(pattern, text, flags=re.I) is not None


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
        row = {"id": c["id"], "split": c["split"], "unanswerable": bool(c.get("unanswerable")), "refused": res["refused"],
               "refused_by": res["refused_by"], "reason": res["reason"], "answer": res["answer"],
               "citations": res["citations"]}
        if not row["unanswerable"] and not res["refused"]:
            row["fact"] = all(has_fact(m, res["answer"]) for m in c["must"])
            row["cited"] = bool(set(res["citations"]) & set(gold_chunk_ids(index, c)))
        rows.append(row)
    return rows


def refused_by(rows):
    """Count refusals by who refused: {"gate": n, "model": n, "extractive": n, "error": n}."""
    return {k: sum(r["refused"] and r["refused_by"] == k for r in rows) for k in REFUSED_BY}


def summarize(rows):
    ans = [r for r in rows if not r["unanswerable"]]
    un = [r for r in rows if r["unanswerable"]]
    given = [r for r in ans if not r["refused"]]
    return {
        "answered": sum(not r["refused"] for r in ans), "n_ans": len(ans),
        "refused": sum(r["refused"] for r in un), "n_un": len(un),
        "fact": sum(r["fact"] for r in given), "cited": sum(r["cited"] for r in given), "n_given": len(given),
        "accuracy": (sum(not r["refused"] for r in ans) + sum(r["refused"] for r in un)) / (len(rows) or 1),
        "unanswerable_refused_by": refused_by(un),
        "answerable_refused_by": refused_by(ans),
    }


def tune_threshold(index, dev):
    scores = {t: summarize(run_answers(index, dev, t))["accuracy"] for t in GRID}
    best = max(scores.values())
    tied = [t for t, a in scores.items() if a == best]
    return tied[len(tied) // 2], scores


def fmt(s):
    return (f"answered {s['answered']}/{s['n_ans']}, refused {s['refused']}/{s['n_un']}, "
            f"fact {s['fact']}/{s['n_given']}, cited {s['cited']}/{s['n_given']}")


def fmt_by(counts, live):
    """Offline there is no model, so show the extractive answerer in its place."""
    keys = ("gate", "model", "error") if live else ("gate", "extractive")
    return ", ".join(f"{k} {counts[k]}" for k in keys)


def result_names(live):
    slug = "".join(ch if ch.isalnum() else "-" for ch in live).strip("-") if live else None
    base = f"live-{slug}" if slug else "offline-latest"
    return base + ".md", base + ".json"


def main(argv=None, out_dir=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="MODEL")
    ap.add_argument("--threshold", type=float)
    a = ap.parse_args(argv)
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
    all_rows = [run_answers(index, x, thr, llm) for x in (dev, test, hold)]
    sd, st, sh = (summarize(r) for r in all_rows)
    flat = [r for rows in all_rows for r in rows]
    misses = [r for r in flat if (r["unanswerable"] and not r["refused"])
              or (not r["unanswerable"] and (r["refused"] or not r.get("fact", True)))]
    live = bool(a.live)
    mode = f"live: {a.live}" if a.live else "offline: BM25 + extractive answers"
    md_name, json_name = result_names(a.live)
    lines = [f"# RAG eval ({mode})", "",
             f"- Corpus: {len(index.chunks)} chunks from {len({c.doc for c in index.chunks})} documents. Cases: {len(dev)} dev, {len(test)} test, {len(hold)} holdout.",
             f"- Refusal threshold: **{thr}** " + ("(given)" if a.threshold is not None else "(tuned on dev only)"),
             f"- Every case, with its answer and the reason for each refusal: `evals/results/{json_name}`", "",
             "| Split | hit@1 | hit@3 | hit@4 | MRR |", "|---|---|---|---|---|",
             f"| dev | {rd['hit1']:.0%} | {rd['hit3']:.0%} | {rd['hit4']:.0%} | {rd['mrr']:.2f} |",
             f"| test | {rt['hit1']:.0%} | {rt['hit3']:.0%} | {rt['hit4']:.0%} | {rt['mrr']:.2f} |",
             f"| holdout | {rh['hit1']:.0%} | {rh['hit3']:.0%} | {rh['hit4']:.0%} | {rh['mrr']:.2f} |", "",
             "| Split | Answer and refusal |", "|---|---|", f"| dev | {fmt(sd)} |", f"| test | {fmt(st)} |", f"| holdout | {fmt(sh)} |", "",
             "Refused by (who refused). An error counts as a refusal above, so a correct refusal is only one made by the gate"
             + (" or the model." if live else " or the extractive answerer."), "",
             "| Split | Unanswerable, refused by | Answerable, refused by |", "|---|---|---|"]
    for name, s in (("dev", sd), ("test", st), ("holdout", sh)):
        lines.append(f"| {name} | {fmt_by(s['unanswerable_refused_by'], live)} | {fmt_by(s['answerable_refused_by'], live)} |")
    lines += ["", "Misses: none" if not misses else "Misses:"]
    for r in misses:
        kind = "answered an unanswerable question" if r["unanswerable"] else ("refused" if r["refused"] else "wrong fact")
        lines.append(f"- `{r['id']}`: {kind}" + (f" ({r['refused_by']}: {r['reason']})" if r["reason"] else ""))
    if scores:
        lines += ["", "Dev accuracy by threshold: " + ", ".join(f"{t}: {v:.0%}" for t, v in scores.items())]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.threshold is None:
        out = out_dir or os.path.join(HERE, "results")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, md_name), "w", encoding="utf-8") as f:
            f.write(text)
        detail = {"mode": mode, "threshold": thr, "chunks": len(index.chunks),
                  "summary": {"dev": sd, "test": st, "holdout": sh}, "cases": flat}
        with open(os.path.join(out, json_name), "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=1, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
