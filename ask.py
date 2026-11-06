"""Ask a question about a folder of .md / .txt documents.

    python ask.py "What was Helio Solar's gross margin?"
    python ask.py --docs path/to/my/notes --model nvidia/nemotron-3-ultra-550b-a55b:free "question"

Without --model the answer is extractive (offline, no key). With --model it calls OpenRouter (needs OPENROUTER_API_KEY).
"""
import argparse
import os

from ragkit import Index, answer
from ragkit.llm import make_llm

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--docs", default=os.path.join(HERE, "corpus", "sample"))
    p.add_argument("--model")
    p.add_argument("--threshold", type=float, default=0.35)
    a = p.parse_args()
    res = answer(a.question, Index.from_folder(a.docs), a.threshold, llm=make_llm(a.model) if a.model else None)
    print(res["answer"])
    if res["citations"]:
        print("Sources:", ", ".join(res["citations"]))
    elif res["reason"]:
        print(f"({res['reason']})")


if __name__ == "__main__":
    main()
