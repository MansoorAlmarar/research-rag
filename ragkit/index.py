"""BM25 index over chunks, plus a coverage score used to decide when to refuse."""
import math
import os
from collections import Counter

from .text import Chunk, chunk_markdown, entities, tokens


class Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(c.text)) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.n = len(self.chunks)
        self.df = df

    @classmethod
    def from_folder(cls, folder, limit=700):
        chunks = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".md", ".txt")):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    chunks += chunk_markdown(name, f.read(), limit)
        if not chunks:
            raise ValueError(f"no .md or .txt documents in {folder}")
        return cls(chunks)

    def idf(self, term):
        d = self.df.get(term, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))

    def score(self, q_terms, i):
        tf, ln = self.tf[i], self.len[i]
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
        return s

    def coverage(self, q_terms, i):
        """Share of the question's idf weight whose terms appear in chunk i. Terms found nowhere in the corpus count
        against it, so a question about something the documents never mention scores low."""
        total = sum(self.idf(t) for t in q_terms)
        if total == 0:
            return 0.0
        return sum(self.idf(t) for t in q_terms if t in self.tf[i]) / total

    def search(self, query, k=4):
        """Top-k chunks as (chunk, bm25, coverage), best first."""
        q = list(dict.fromkeys(tokens(query)))
        if not q or not self.chunks:
            return []
        ents = entities(query)
        # A question that names a company only matches chunks about that company. A name found nowhere matches nothing.
        pool = [i for i in range(self.n) if any(e in self.tf[i] for e in ents)] if ents else range(self.n)
        scored = sorted(((self.score(q, i), i) for i in pool), reverse=True)[:k]
        return [(self.chunks[i], s, self.coverage(q, i)) for s, i in scored]
