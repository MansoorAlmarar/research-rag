"""Ask a question about a folder of .md / .txt documents.

    python ask.py "What was Helio Solar's gross margin?"
    python ask.py --docs path/to/my/notes --model nvidia/nemotron-3-ultra-550b-a55b:free "question"

Without --model the answer is extractive (offline, no key). With --model it calls OpenRouter (needs OPENROUTER_API_KEY).
"""
import argparse
import os
import sys

from ragkit import DEFAULT_THRESHOLD, Index, answer
from ragkit.llm import make_llm

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--docs", default=os.path.join(HERE, "corpus", "sample"))
    p.add_argument("--model")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = p.parse_args(argv)
    try:
        index = Index.from_folder(a.docs)
        llm = make_llm(a.model) if a.model else None
    except (OSError, ValueError, RuntimeError) as e:
        # A missing folder, no documents, a missing key or a missing CLI: one line, not a traceback.
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = answer(a.question, index, a.threshold, llm=llm)
    print(res["answer"])
    if res["citations"]:
        print("Sources:", ", ".join(res["citations"]))
    elif res["reason"]:
        print(f"({res['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
