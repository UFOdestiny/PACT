"""Reward functions for the GRPO stack, backed by the evaluation judge.

``training/entrypoint.py`` registers them as ``format``, ``accuracy``, ``brier``, ``mean_confidence`` and
``confidence_one_or_zero``.  They keep RLCR's semantics (format gate first, Brier on the verbal confidence) but
verify correctness with ``evaluation.tasks`` -- the cascade the evaluation uses -- instead of RLCR's bare
``math_verify`` call, which parsed the stored gold as free text and produced 7-8 % false negatives on
MATH-style LaTeX golds.  ``correctness_labels`` and ``answer_identities`` give the trainer the same verdicts
and the judge-consistent answer identity of every rollout.
"""

from __future__ import annotations

import re

from evaluation import tasks
from evaluation.generate import format_compliant, parse_answer, parse_confidence


def _contents(completions) -> list[str]:
    return [completion[0]["content"] for completion in completions]


def _dataset(source) -> str:
    """Training rows carry the pool's own ``source`` (orca_math, aops, ...); every
    such row is a math problem.  Only evaluation dataset names route elsewhere."""
    source = str(source or "")
    if source in tasks.CHOICE_DATASETS or source in tasks.SHORT_ANSWER_DATASETS:
        return source
    return "math"


# RLVR and RLCR keep their published hard format gate on the accuracy reward.
GATED = ("rlcr", "reasoning_answer")


def format_reward(format_pattern, completions, **kwargs) -> list[float]:
    rewards = []
    for content in _contents(completions):
        if not format_compliant(content, format_pattern):
            rewards.append(0.0)
            continue
        if format_pattern == "rlcr" and parse_confidence(content, style="tag", allow_percent=False) is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0)
    return rewards


def accuracy_reward(format_pattern, completions, answer, source=None, **kwargs) -> list[float]:
    """Verified correctness.  RLVR and RLCR keep their published hard format gate (official
    reward_fns.accuracy_reward scores 0 whenever format_reward is 0); the boxed/line schemas do not."""
    formats = format_reward(format_pattern, completions)
    rewards = []
    sources = source if source is not None else [None] * len(answer)
    for content, gold, src, ok in zip(_contents(completions), answer, sources, formats, strict=True):
        if format_pattern in GATED and not ok:
            rewards.append(0.0)
            continue
        dataset = _dataset(src)
        raw = parse_answer(content)
        prediction = None if raw is None else tasks.parse(dataset, raw, None)
        rewards.append(float(tasks.judge(dataset, prediction, str(gold), None)))
    return rewards


def brier_reward(format_pattern, completions, answer, source=None, **kwargs) -> list[float]:
    """RLCR's ``1 - (correct - confidence)^2`` on the verbal confidence, format gated."""
    formats = format_reward(format_pattern, completions)
    correctness = accuracy_reward(format_pattern, completions, answer, source)
    rewards = []
    for content, y, ok in zip(_contents(completions), correctness, formats, strict=True):
        confidence = parse_confidence(content, style="tag", allow_percent=False) if ok else None
        rewards.append(0.0 if confidence is None else 1.0 - (y - confidence) ** 2)
    return rewards


def _confidences(completions) -> list[float | None]:
    values = []
    for content in _contents(completions):
        matches = re.findall(r"<confidence>(.*?)</confidence>", content, re.DOTALL | re.MULTILINE)
        try:
            values.append(max(0.0, min(float(matches[-1]), 1.0)) if matches else None)
        except ValueError:
            values.append(None)
    return values


def mean_confidence_reward(completions, answer, **kwargs) -> list[float]:
    return [0.0 if c is None else c for c in _confidences(completions)]


def confidence_one_or_zero(completions, answer, **kwargs) -> list[float]:
    return [1.0 if c is not None and (abs(c - 1.0) < 0.01 or abs(c) < 0.01) else 0.0
            for c in _confidences(completions)]


def correctness_labels(texts: list[str], golds: list, sources: list) -> list[bool]:
    """Answer correctness independent of a method's formatting contract."""
    labels = []
    for text, gold, src in zip(texts, golds, sources, strict=True):
        raw = parse_answer(text)
        prediction = None if raw is None else tasks.parse(_dataset(src), raw, None)
        labels.append(bool(tasks.judge(_dataset(src), prediction, str(gold), None)))
    return labels


def answer_identities(texts: list[str], sources: list, *, group_width: int | None = None) -> list[str]:
    """Judge-consistent answer identity per rollout (``evaluation.identity``).

    Clustering is per rollout group: two responses to *different* prompts are
    never the same answer, and the symbolic cascade only ever compares the
    ``group_width`` responses that share a prompt.  ``group_width=None``
    clusters the whole list, which is what a single group's worth of texts
    needs.
    """
    from evaluation.identity import cluster

    parsed = []
    for text, src in zip(texts, sources, strict=True):
        raw = parse_answer(text)
        parsed.append(None if raw is None else tasks.parse(_dataset(src), raw, None))
    dataset = _dataset(sources[0] if sources else None)
    width = len(parsed) if group_width is None else group_width
    if width <= 0 or len(parsed) % width:
        raise ValueError("texts must contain complete rollout groups")
    keys: list[str] = []
    for start in range(0, len(parsed), width):
        block = cluster(parsed[start:start + width], list(range(start, start + width)), dataset)
        keys.extend(block)
    return keys
