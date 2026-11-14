import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

import server  # noqa: E402
from ragkit import Index  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "corpus", "sample")
INDEX = Index.from_folder(SAMPLE)


def test_answer_has_sources_with_text():
    out = server.run_question("How many vessels does Tidewater run?", INDEX)
    assert not out["refused"]
    assert out["sources"] and out["sources"][0]["text"]
    assert "214" in out["answer"]


def test_unanswerable_question_is_refused_with_reason():
    out = server.run_question("Who is the chief financial officer of Tidewater Logistics?", INDEX)
    assert out["refused"] and out["sources"] == [] and out["reason"]


def test_empty_and_long_questions_return_an_error():
    assert "error" in server.run_question("   ", INDEX)
    assert "error" in server.run_question("x" * 301, INDEX)
