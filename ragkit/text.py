"""Tokenising and chunking. No dependencies."""
import re
from dataclasses import dataclass

STOP = frozenset("""a an and are as at be by did do does for from had has have how in is it its of on or per than that the
their there this to was were what when where which who whom whose why will with would about after before during into over
under up down out off not no than then them they he she we you i my our your his her
much many long big large come number amount""".split())

TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")


def stem(w):
    """Very light stemmer: enough for plural and -ed / -ing forms, no more."""
    if w[0].isdigit() or len(w) <= 4:
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


POSSESSIVE = re.compile(r"['’]s\b")
TITLE_NAME = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")


def title_names(title):
    """Capitalised words in a document title, lowercased and stemmed: "Tidewater Logistics, annual report" gives
    tidewater and logistic. The index turns these into the list of names a question can be about (see Index)."""
    words = TITLE_NAME.findall(POSSESSIVE.sub("", title))
    return list(dict.fromkeys(stem(w.lower()) for w in words if w.lower() not in STOP))


CAPITALISED = re.compile(r"\b[A-Z][a-z]{2,}\b")


def capitalised_words(question):
    """Capitalised words after the first word, lowercased and stemmed: likely names, such as France or Bloomberg.
    Only a safety net (see Index.search). Which company a question is about is decided by the corpus, not by capitals."""
    words = list(CAPITALISED.finditer(POSSESSIVE.sub("", question)))
    return list(dict.fromkeys(stem(m.group().lower()) for m in words if m.start() > 0 and m.group().lower() not in STOP))


def tokens(text, keep_stop=False):
    out = []
    for t in TOKEN.findall(POSSESSIVE.sub("", text.lower())):
        t = t.replace(",", "")
        if not keep_stop and t in STOP:
            continue
        out.append(stem(t))
    return out


@dataclass(frozen=True)
class Chunk:
    id: str        # "<file stem>#<n>", what the answer cites
    doc: str       # file name
    heading: str   # section heading, "" for the intro
    text: str      # chunk body (what the model sees)
    title: str = ""  # document title (the `#` heading), "" if the file has none


def _split_long(body, limit):
    """Split a long section at sentence ends so no chunk is much longer than `limit` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_markdown(doc_name, text, limit=700):
    """One chunk per section (a `#` or `##` heading starts a section); long sections are split at sentence ends."""
    stem_name = re.sub(r"\.[^.]+$", "", doc_name)
    title, heading, body, sections = "", "", [], []
    for line in text.splitlines():
        m = re.match(r"^(#{1,2})\s+(.*)", line)
        if m:
            if body:
                sections.append((heading, "\n".join(body).strip()))
            body = []
            if len(m.group(1)) == 1:
                title, heading = m.group(2).strip(), ""
            else:
                heading = m.group(2).strip()
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body).strip()))
    chunks = []
    for h, b in sections:
        if not b:
            continue
        for part in _split_long(b, limit):
            label = f"{title} / {h}" if h else title
            chunks.append(Chunk(f"{stem_name}#{len(chunks)}", doc_name, h, f"{label}\n{part}", title))
    return chunks
