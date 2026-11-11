"""Answer identity for the consensus readout and for agreement-based credit.

Two responses share an identity when they give the same answer.  For
multiple-choice and short-answer sets ``tasks.parse`` already canonicalises the
answer, so identity is the normalised string.  For math the string identity
splits ``\\frac{1}{2}``, ``0.5`` and ``1/2`` into three answers while the judge
marks all three correct, which under-states agreement for methods that vary
their surface form. ``cluster`` therefore groups math answers with the same cascade the
judge uses (exact -> numeric -> symbolic) and never reads the gold answer, so
it stays a legitimate gold-free confidence signal.

An unparseable response holds a sample-specific identity: it agrees with
nothing and nothing agrees with it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from evaluation import tasks


def invalid_identity(sample_id: int) -> str:
    return f"<INVALID:{int(sample_id)}>"


def is_invalid(identity: str) -> bool:
    return identity.startswith("<INVALID:")


def string_identity(parsed: Any, sample_id: int) -> str:
    """Whitespace/NFC-normalised answer string; NFKC is avoided on purpose because
    it maps ``130₄`` to ``1304`` and would merge distinct answers."""
    if parsed is None or not str(parsed).strip():
        return invalid_identity(sample_id)
    value = unicodedata.normalize("NFC", str(parsed)).strip()
    value = re.sub(r"\s+", "", value).strip("$")
    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value or invalid_identity(sample_id)


def cluster(parsed_answers: list[Any], sample_ids: list[int], dataset: str) -> list[str]:
    """Identity label per response.  On math datasets answers the judging cascade would call equivalent are
    merged; elsewhere the normalised string is the identity.  Labels are the canonical string of the cluster's
    first member so they stay readable in the bank."""
    labels = [string_identity(p, s) for p, s in zip(parsed_answers, sample_ids, strict=True)]
    if dataset in tasks.CHOICE_DATASETS or dataset in tasks.SHORT_ANSWER_DATASETS or dataset in tasks.CODE_DATASETS:
        return labels
    representatives: list[tuple[str, str]] = []   # (label, raw canonical answer)
    readings = tasks.Readings()                    # parsed answers, shared by every comparison below
    decided: list[tuple[str, str, bool]] = []      # (raw, representative raw, verdict) already judged

    def equivalent(raw: str, existing_raw: str) -> bool:
        # A repeated answer walks the same representatives; judge each ordered pair once.
        for seen_raw, seen_existing, verdict in decided:
            if seen_raw == raw and seen_existing == existing_raw:
                return verdict
        before = tasks.timeouts()
        verdict = tasks.math_equivalent_with_stage(raw, existing_raw, readings)[0]
        if tasks.timeouts() == before:
            decided.append((raw, existing_raw, verdict))
        return verdict

    out: list[str] = []
    for parsed, label in zip(parsed_answers, labels, strict=True):
        if is_invalid(label):
            out.append(label)
            continue
        raw = str(parsed)
        found = None
        for existing_label, existing_raw in representatives:
            if existing_label == label or equivalent(raw, existing_raw):
                found = existing_label
                break
        if found is None:
            representatives.append((label, raw))
            found = label
        out.append(found)
    return out
