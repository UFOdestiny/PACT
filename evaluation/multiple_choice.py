"""Lenient multiple-choice judge: the paper's labels for ARC-Challenge (and the other letter-answer sets).

Answer letter, first rule that fires:
  1. the stored strict parse (single letter, or exact option text) - what the live judge accepted;
  2. the boxed content starts with an option letter followed by . ) : or a space (also inside \\text{});
  3. the boxed content equals an option text after normalisation, or exactly one option's text contains it / is
     contained in it (the answer >= 4 characters inside an option, or an option >= 8 characters inside the answer);
  4. without a usable boxed answer: the last "final answer is X" / "answer: X" phrase in the response outside the
     confidence report lines;
  5. the last non-report line: a lone letter, "option X", or a unique option-text match as in 3 ("option X" inside the
     reasoning is as often a rejection, so it only counts on the last line).
"""
from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJ"
_PREFIX = re.compile(r"^\s*(?:\\text\{\s*)?\(?([A-J])\)?(?:[.):]|\s|$)")
_PHRASE = re.compile(r"(?i)(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)?\s*[:\-]?\s*\**\s*(?:option\s*)?\(?\s*([A-J])\s*\)?(?![A-Za-z0-9])")
_OPTION = re.compile(r"(?i)\boption\s*\(?\s*([A-J])\s*\)?(?![A-Za-z0-9])")
REPORT_LINE = re.compile(r"(?i)^\s*(?:confidence|agreement)\s*:")


def norm(text: str) -> str:
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", str(text))
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return " ".join(re.sub(r"[^0-9a-z.%-]+", " ", text.lower()).split()).strip(" .")


def option_match(text: str, choices: list[str]) -> str | None:
    t = norm(text)
    if not t:
        return None
    normed = [norm(c) for c in choices]
    exact = [i for i, c in enumerate(normed) if c and c == t]
    if len(exact) == 1:
        return LETTERS[exact[0]]
    if len(t) >= 4:
        # the answer inside an option text, or a long option text (>= 8 chars) inside the answer; short options such as
        # "none" are only matched exactly
        within = [i for i, c in enumerate(normed) if c and ((t in c) or (len(c) >= 8 and c in t))]
        if len(within) == 1:
            return LETTERS[within[0]]
    return None


def lenient_letter(row: dict, choices: list[str]) -> str | None:
    n = len(choices)
    valid = lambda letter: letter if letter and LETTERS.index(letter) < n else None
    if row.get("answer_parse_valid") and str(row.get("parsed_answer") or "").strip().upper() in LETTERS[:n]:
        return str(row["parsed_answer"]).strip().upper()
    boxed = str(row.get("parsed_answer") or "").strip()
    if boxed:
        m = _PREFIX.match(boxed)
        if m and valid(m.group(1)):
            return m.group(1)
        hit = option_match(boxed, choices)
        if hit:
            return hit
    body = [ln for ln in str(row.get("response") or "").splitlines() if ln.strip() and not REPORT_LINE.match(ln)]
    text = "\n".join(body)
    found = [m.group(1).upper() for m in _PHRASE.finditer(text) if valid(m.group(1).upper())]
    if found:
        return found[-1]
    if body:
        # "option X" only on the last line: inside the reasoning it is as often a rejection ("option E is out")
        last = body[-1].strip().strip("*").strip()
        m = re.fullmatch(r"\(?([A-J])\)?[.)]?", last) or _OPTION.search(last)
        if m and valid(m.group(1).upper()):
            return m.group(1).upper()
        return option_match(last, choices)
    return None
