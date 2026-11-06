import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))

from ragkit import REFUSAL, Index, answer  # noqa: E402
from ragkit.answer import check_model_answer  # noqa: E402
from ragkit.text import chunk_markdown, entities, tokens  # noqa: E402

import run as ev  # noqa: E402

CORPUS = os.path.join(ROOT, "corpus", "sample")


@pytest.fixture(scope="module")
def index():
    return Index.from_folder(CORPUS)


# ---------- text ----------
def test_tokens_drop_possessive_and_question_words():
    assert tokens("What was Helio's gross margin?") == tokens("Helio gross margin")
    assert "much" not in tokens("How much revenue")


def test_numbers_stay_whole():
    assert "31.4" in tokens("Gross margin was 31.4%")
    assert "1580" in tokens("between $1,580 and")


def test_entities_are_names_after_the_first_word():
    assert entities("Who is the chief financial officer of Tidewater Logistics?") == ["tidewater", "logistic"]
    assert entities("What was Helio's net income?") == ["helio"]
    assert entities("Revenue grew") == []


def test_chunks_follow_sections_and_keep_the_title():
    text = "# Acme report\n\nIntro line here.\n\n## Costs\n\nFuel rose. Wages rose.\n\n## Risks\n\nA strike hit output.\n"
    chunks = chunk_markdown("acme.md", text)
    assert [c.heading for c in chunks] == ["", "Costs", "Risks"]
    assert chunks[1].id == "acme#1" and "Acme report / Costs" in chunks[1].text


def test_long_sections_split_at_sentence_ends():
    body = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_markdown("long.md", f"# T\n\n## S\n\n{body}\n", limit=300)
    assert len(chunks) > 3
    assert all(len(c.text) < 400 and c.text.rstrip().endswith(".") for c in chunks)


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        Index.from_folder(str(tmp_path))


# ---------- retrieval ----------
def test_right_chunk_is_first(index):
    top = index.search("How many vessels does Tidewater run?")[0][0]
    assert top.doc.startswith("tidewater") and "214 vessels" in top.text


def test_a_named_company_keeps_results_on_that_company(index):
    hits = index.search("Who is the chief financial officer of Tidewater Logistics?")
    assert hits and all(c.doc.startswith("tidewater") for c, _s, _v in hits)


def test_an_unknown_name_matches_nothing(index):
    assert index.search("What is the capital of France?") == []


# ---------- answers ----------
def test_answerable_question_is_answered_with_a_citation(index):
    res = answer("How many vessels does Tidewater run?", index)
    assert not res["refused"] and "214" in res["answer"] and res["citations"]


def test_unanswerable_question_is_refused(index):
    res = answer("What is the capital of France?", index)
    assert res["refused"] and res["answer"] == REFUSAL


def _hits(index, q):
    return index.search(q)


def test_model_answer_needs_a_retrieved_citation(index):
    h = _hits(index, "How many vessels does Tidewater run?")
    ok, cites, why = check_model_answer({"answer": "It runs 214 vessels.", "citations": [h[0][0].id]}, h)
    assert why is None and cites == [h[0][0].id]
    assert check_model_answer({"answer": "It runs 214 vessels.", "citations": ["made-up#9"]}, h)[2] == "no valid citation"
    assert check_model_answer({"answer": "It runs 214 vessels."}, h)[2] == "no valid citation"


def test_model_answer_with_an_invented_number_is_dropped(index):
    h = _hits(index, "How many vessels does Tidewater run?")
    _a, _c, why = check_model_answer({"answer": "It runs 999 vessels.", "citations": [h[0][0].id]}, h)
    assert why.startswith("number 999")


@pytest.mark.parametrize("reply", ["not json at all", "[]", '{"answer": ""}', {"answer": None}, 42])
def test_bad_model_replies_become_refusals_not_crashes(index, reply):
    res = answer("How many vessels does Tidewater run?", index, llm=lambda s, u: reply)
    assert res["refused"] and res["reason"]


def test_model_can_say_not_in_documents(index):
    res = answer("How many vessels does Tidewater run?", index, llm=lambda s, u: '{"answer": "NOT_IN_DOCUMENTS", "citations": []}')
    assert res["refused"] and res["reason"] == "model said not in documents"


def test_model_failure_is_a_refusal(index):
    def boom(s, u):
        raise RuntimeError("HTTP 503")
    res = answer("How many vessels does Tidewater run?", index, llm=boom)
    assert res["refused"] and res["reason"] == "model error: RuntimeError"


def test_prompt_injection_text_in_a_source_is_sent_as_quoted_source(index):
    seen = {}

    def spy(system, user):
        seen["system"], seen["user"] = system, user
        return {"answer": "NOT_IN_DOCUMENTS", "citations": []}
    answer("How many vessels does Tidewater run?", index, llm=spy)
    assert "untrusted" in seen["system"] and "SOURCES:" in seen["user"]


# ---------- the eval itself ----------
def test_every_gold_phrase_exists_in_the_corpus(index):
    for c in ev.load_cases():
        if not c.get("unanswerable"):
            assert ev.gold_chunk_ids(index, c), c["id"]


def test_case_ids_are_unique_and_splits_known():
    cases = ev.load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["split"] for c in cases} == {"dev", "test", "holdout"}


def test_offline_eval_does_not_regress(index):
    """Floors set a little under the numbers in the README, so a real regression fails and noise does not."""
    cases = ev.load_cases()
    r = ev.retrieval_stats(index, cases)
    assert r["hit3"] >= 0.95 and r["mrr"] >= 0.9
    dev = [c for c in cases if c["split"] == "dev"]
    thr, _ = ev.tune_threshold(index, dev)
    s = ev.summarize(ev.run_answers(index, cases, thr))
    assert s["answered"] >= 28 and s["refused"] >= 10
    assert s["cited"] == s["n_given"]
