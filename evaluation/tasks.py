"""Dataset-specific answer canonicalisation and the judging cascade.

Math answers go through a fixed cascade -- exact string, numeric value,
symbolic verifier -- and every verdict records which stage decided it.
Multiple-choice sets answer with an option letter, short-answer sets are judged
by normalised exact match against the gold alias set, code sets by executing the
answer against the set's tests in a sandboxed subprocess (``evaluation/code_exec.py``).  ``parse`` turns the raw
``<answer>`` text into the canonical string that both the judge and the
consensus identity see.

The same functions are the training reward's verifier (``training/rewards.py``)
and the evaluation judge (``evaluation/generate.py``). Parsing and format
gating can still differ between reward and evaluation events.  No LLM judge is
used, in training or in evaluation.
"""

from __future__ import annotations

import logging
import re
import signal
import sys
import threading
from fractions import Fraction

from math_verify import parse as mv_parse, verify as mv_verify
from sympy import srepr

CHOICE_DATASETS = {"arc_challenge"}
SHORT_ANSWER_DATASETS = {"simpleqa"}
# Function-completion sets judged by execution (tools/prepare_eval_data.py); the answer is the program text.
CODE_DATASETS = {"mbpp_plus_fenced"}
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_PUNCTUATION = re.compile(r"[^\w\s]")
_NUMERIC_TOLERANCE = Fraction(1, 1000000)


def normalize_short_answer(value: str) -> str:
    """Lowercase, drop articles and punctuation, collapse whitespace.

    The standard HotpotQA/SQuAD normalisation.  It is the answer identity as
    well as the judging key, so "The Beatles", "beatles" and "beatles." are one
    answer for the consensus readout, not three.
    """
    text = " ".join(_PUNCTUATION.sub(" ", value.casefold()).split())
    without_articles = " ".join(_ARTICLES.sub(" ", text).split())
    # "A" is a real answer, not an article: only drop articles when something
    # survives, or a one-word answer would become unparseable.
    return without_articles or text


def parse(dataset: str, raw: str, row: dict | None) -> str | None:
    """Canonical answer string for one dataset, or None when unparseable."""
    value = raw.strip()
    if not value:
        return None
    if dataset in CODE_DATASETS:
        return raw.strip("\n")
    if dataset in SHORT_ANSWER_DATASETS:
        return normalize_short_answer(value) or None
    if dataset in CHOICE_DATASETS:
        match = re.fullmatch(r"(?i)\s*(?:option\s*)?[\(\[]?([A-J])[\)\].:]?\s*", value)
        if match:
            return match.group(1).upper()
        for index, choice in enumerate(list((row or {}).get("choices") or [])):
            if value.casefold() == str(choice).strip().casefold():
                return chr(ord("A") + index)
        return None
    return value


def judge(dataset: str, prediction: str | None, gold: str, row: dict | None) -> bool:
    """Response-level correctness of a canonical answer against the stored gold."""
    return judge_with_stage(dataset, prediction, gold, row)[0]


def judge_with_stage(
    dataset: str, prediction: str | None, gold: str, row: dict | None,
    readings: "Readings | None" = None,
) -> tuple[bool, str]:
    """Correctness plus the cascade stage that decided it (see ``STAGES``)."""
    if prediction is None:
        return False, "unparsed"
    if dataset in CODE_DATASETS:
        from evaluation import code_exec

        return code_exec.judge(prediction, row)
    if dataset in SHORT_ANSWER_DATASETS:
        predicted = normalize_short_answer(prediction)
        # A short-answer set may list every accepted spelling of the fact (``aliases``; SimpleQA has one).
        # ``aliases`` arrives as a list from parquet and as a numpy array from
        # pandas; never truth-test it, only its length.
        raw = row.get("aliases") if row is not None else None
        aliases = [str(alias) for alias in (raw if raw is not None else [])]
        accepted = aliases if len(aliases) else [str(gold)]
        hit = any(predicted == normalize_short_answer(str(alias)) for alias in accepted)
        return hit, "exact" if hit else "reject"
    if dataset in CHOICE_DATASETS:
        predicted, expected = prediction.strip().upper(), str(gold).strip().upper()
        hit = bool(re.fullmatch(r"[A-J]", predicted) and predicted == expected)
        return hit, "exact" if hit else "reject"
    return math_equivalent_with_stage(prediction, gold, readings)


# --- math cascade -----------------------------------------------------------

_STRIP_PATTERNS = (
    (re.compile(r"\\(?:left|right|,|!|;|:|quad|qquad)"), ""),
    (re.compile(r"\\(?:d|t)frac"), r"\\frac"),
    (re.compile(r"\\text\{\s*\}"), ""),
    (re.compile(r"\^\{?\\circ\}?|°"), ""),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"\\\$|\$"), ""),
    (re.compile(r"\s+"), ""),
)


def strip_math(value: str) -> str:
    """Hendrycks-style normalisation for the exact-match stage: drop spacing
    macros, thin-space thousands separators, degree/percent/dollar signs and
    whitespace; unify ``\\dfrac``/``\\tfrac`` with ``\\frac``."""
    out = str(value).strip()
    for pattern, replacement in _STRIP_PATTERNS:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", out)
    out = re.sub(r"\\boxed\{(.*)\}$", r"\1", out)
    if re.fullmatch(r"[-+]?\d+\.0+", out):
        out = out.split(".", 1)[0]
    return out


_FRACTION = re.compile(r"^([-+])?\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}$")
_SLASH = re.compile(r"^([-+]?\d+)/([-+]?\d+)$")
_NUMBER = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def _too_many_digits(*parts: str) -> bool:
    """Whether ``int()`` would refuse these digit strings.

    CPython caps integer <-> string conversion at ``sys.get_int_max_str_digits()``
    (4,300 by default) and raises ``ValueError`` past it.  A policy model can emit a
    number far longer than that -- one Llama-3.1-8B baseline produced a 12,288-digit answer on
    MATH-500 -- and an uncaught raise takes down the whole evaluation shard.  The
    threshold is Python's own limit on purpose: every string this function could
    previously convert still converts, so no already-scored response changes; only
    inputs that used to crash now become unparseable, which is the existing
    treatment for an answer we cannot read.
    """
    limit = sys.get_int_max_str_digits()
    return limit > 0 and any(sum(c.isdigit() for c in part) > limit for part in parts if part)


def to_number(value: str) -> Fraction | None:
    """Exact rational value of a plain number, fraction or ``a/b``; None otherwise.

    Thousands separators (``10,080`` and the LaTeX ``10,\\!080``) are accepted
    only in the canonical 3-digit grouping so that ``1,2`` stays a tuple.
    """
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", strip_math(value))
    if _NUMBER.match(text):
        digits = text.replace(",", "")
        return None if _too_many_digits(digits) else Fraction(digits)
    match = _FRACTION.match(text)
    if match:
        sign, numerator, denominator = match.groups()
        if _too_many_digits(numerator, denominator) or int(denominator) == 0:
            return None
        result = Fraction(int(numerator), int(denominator))
        return -result if sign == "-" else result
    match = _SLASH.match(text)
    if match:
        numerator, denominator = match.group(1), match.group(2)
        if _too_many_digits(numerator, denominator) or int(denominator) == 0:
            return None
        return Fraction(int(numerator), int(denominator))
    return None


def _math_parse(value: str):
    """math_verify expects LaTeX with delimiters.  Fed a bare string such as
    ``\\frac 34`` or ``10,\\!080`` it falls back to grabbing the first number
    (34, 10), which produced both false negatives and false positives.  Parse
    the value as ``$...$`` and as ``\\boxed{...}`` and keep every reading."""
    readings = []
    for wrapped in (f"${value}$", f"\\boxed{{{value}}}"):
        try:
            parsed = mv_parse(wrapped, parsing_timeout=2)
        except Exception:
            parsed = None
        if parsed:
            readings.append(parsed)
    return readings


class _Timeouts(logging.Handler):
    """Counts math_verify's wall-clock timeouts (it logs one warning per timed-out parse or comparison and
    returns an empty / False result).  A result computed across a timeout is never reused, so a repeat
    is recomputed exactly as the uncached cascade would recompute it."""

    count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if str(record.msg).startswith("Timeout during"):
            _Timeouts.count += 1
        logging.lastResort.handle(record)   # keep the warning on stderr as before


logging.getLogger("math_verify").addHandler(_Timeouts())


def timeouts() -> int:
    return _Timeouts.count


class _ShapeTimeout(Exception):
    pass


def _shape(reading) -> list | None:
    """``srepr`` of each parsed item, or None.  ``order='none'`` prints arguments in their stored order: the
    default ordering of an ``Add`` evaluates its terms numerically, and a ``Sum`` term can then run
    Euler-Maclaurin without end (one judge sat in it for an hour).  A one-second alarm
    bounds the rest; None only disables the rejected-pair skip, so the verdict is unchanged."""
    if threading.current_thread() is not threading.main_thread():   # signals only reach the main thread
        try:
            return [srepr(item, order="none") for item in reading]
        except Exception:
            return None

    def expire(signum, frame):
        raise _ShapeTimeout
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        return [srepr(item, order="none") for item in reading]
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Readings:
    """``_math_parse`` per answer string for one problem, found by ``==`` in a list (no hashing), each
    reading with its ``srepr`` shape (None when it has none)."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._values: list[list] = []

    def __call__(self, value: str) -> list:
        for key, readings in zip(self._keys, self._values):
            if key == value:
                return readings
        before = timeouts()
        readings = []
        for reading in _math_parse(value):
            shape = _shape(reading)
            readings.append((reading, shape))
        if timeouts() == before:
            self._keys.append(value)
            self._values.append(readings)
        return readings


def math_equivalent_with_stage(prediction: str, gold: str,
                               readings: Readings | None = None) -> tuple[bool, str]:
    """Exact string -> exact rational value -> symbolic verifier -> reject.

    With ``readings`` (one problem's parse memo) the symbolic stage reuses parsed answers and skips a
    reading pair whose shapes equal a pair both verifier directions already rejected without a timeout;
    the verdict is the one the uncached loop returns."""
    prediction, gold = str(prediction).strip(), str(gold).strip()
    if strip_math(prediction) == strip_math(gold):
        return True, "exact"
    predicted_number, gold_number = to_number(prediction), to_number(gold)
    if predicted_number is not None and gold_number is not None:
        # Exact rational arithmetic throughout: a response may hold a number far outside the
        # float range (10^400 is a real rollout), and converting it to float raises OverflowError.
        difference = abs(predicted_number - gold_number)
        scale = max(Fraction(1), abs(gold_number))
        if predicted_number == gold_number or difference <= _NUMERIC_TOLERANCE * scale:
            return True, "numeric"
        # Two well-formed numbers that differ are a definite rejection; the
        # symbolic verifier cannot rescue "3" against "4".
        return False, "reject"
    if readings is not None:
        return _symbolic(readings(gold), readings, prediction)
    for parsed_gold in _math_parse(gold):
        for parsed_prediction in _math_parse(prediction):
            try:
                if mv_verify(parsed_gold, parsed_prediction, timeout_seconds=2) or mv_verify(
                    parsed_prediction, parsed_gold, timeout_seconds=2
                ):
                    return True, "symbolic"
            except Exception:
                continue
    return False, "reject"


def _symbolic(gold_readings: list, readings: Readings, prediction: str) -> tuple[bool, str]:
    prediction_readings = readings(prediction) if gold_readings else []
    rejected: list[tuple[list, list]] = []
    for parsed_gold, gold_shape in gold_readings:
        for parsed_prediction, prediction_shape in prediction_readings:
            if gold_shape is not None and prediction_shape is not None and (gold_shape, prediction_shape) in rejected:
                continue
            before = timeouts()
            try:
                if mv_verify(parsed_gold, parsed_prediction, timeout_seconds=2) or mv_verify(
                    parsed_prediction, parsed_gold, timeout_seconds=2
                ):
                    return True, "symbolic"
            except Exception:
                continue
            if timeouts() == before:
                rejected.append((gold_shape, prediction_shape))
    return False, "reject"


def math_equivalent(prediction: str, gold: str) -> bool:
    """Symmetric equivalence of two math answer strings."""
    return math_equivalent_with_stage(prediction, gold)[0]
