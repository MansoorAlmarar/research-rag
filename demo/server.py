"""Local demo page for research-rag. Standard library only.

    python demo/server.py                      # offline answers, no key
    python demo/server.py --model claude-cli:haiku   # model answers (same as ask.py --model)

Opens http://127.0.0.1:8000. It binds to localhost only, so nobody else can use your model quota.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ragkit import DEFAULT_THRESHOLD, Index, answer  # noqa: E402
from ragkit.llm import make_llm  # noqa: E402

MAX_QUESTION = 300


def run_question(question, index, llm=None):
    """One question in, a JSON-ready dict out: answer, refused, reason, and the cited passages."""
    question = (question or "").strip()
    if not question:
        return {"error": "Type a question first."}
    if len(question) > MAX_QUESTION:
        return {"error": f"Keep the question under {MAX_QUESTION} characters."}
    res = answer(question, index, DEFAULT_THRESHOLD, llm=llm)
    by_id = {c.id: c for c in index.chunks}
    sources = [{"id": cid, "doc": by_id[cid].doc, "title": by_id[cid].title, "text": by_id[cid].text}
               for cid in res["citations"] if cid in by_id]
    return {"answer": res["answer"], "refused": res["refused"], "refused_by": res["refused_by"],
            "reason": res["reason"], "sources": sources}


def make_handler(index, llm):
    page = open(os.path.join(HERE, "index.html"), "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/ask":
                return self._send(404, b"not found", "text/plain")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 4096)
                data = json.loads(self.rfile.read(n) or b"{}")
                out = run_question(data.get("question"), index, llm)
            except (ValueError, TypeError):
                out = {"error": "Bad request."}
            self._send(200, json.dumps(out).encode(), "application/json")

        def log_message(self, *_):  # keep the console quiet
            pass

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=os.path.join(ROOT, "corpus", "sample"))
    p.add_argument("--model")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    try:
        index = Index.from_folder(a.docs)
        llm = make_llm(a.model) if a.model else None
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(index, llm))
    print(f"Demo at http://127.0.0.1:{a.port}  ({'model: ' + a.model if a.model else 'offline answers'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
