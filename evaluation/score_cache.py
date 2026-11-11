"""Per-bank score cache: compact per-response arrays and summary metrics written next to each test bank.

Every analysis used to re-parse ``native.jsonl`` (about 190 MB per bank, mostly token log-probabilities).  ``load``
returns the arrays from ``<bank>/scores.npz`` and rebuilds it only when ``native.jsonl`` (or ``llm_judge.jsonl``) has
changed since it was written; staleness is decided by file size and modification time (no hashing).  The same pass
writes ``<bank>/summary.json`` with accuracy, Brier, ECE and AUROC under every judge label the bank has, with the
paper's definitions (missing confidence scored 0, 10-bin ECE).

Arrays (one row per response, sorted by problem then sample): ``problem_id`` (str), ``sample_id``, ``q`` (stated
confidence, NaN when missing), ``q_valid``, ``answer_parse_valid``, ``truncated``, and correctness labels ``y_strict``
(the stored judge), ``y_lenient`` (multiple choice: evaluation.multiple_choice's lenient letter judge), ``y_llm`` / ``y_llm_strict``
(evaluation/llm_judge.py labels, lenient / strict extraction).  A label is absent when it does not apply to the bank.

    python -m evaluation.score_cache [--root artifacts] [--model llama-3.1-8b-instruct ...]   # backfill complete banks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation.multiple_choice import lenient_letter
from training.experiment import ARTIFACTS, RESOURCE

MC_SETS = ("arc_challenge",)
_CHOICES: dict[str, dict[str, list[str]]] = {}


def ece(c, y, bins=10):
    """10-bin equal-width expected calibration error (the paper's definition)."""
    b = np.clip((np.round(c, 6) * bins).astype(int), 0, bins - 1)
    return sum(np.mean(b == k) * abs(c[m := b == k].mean() - y[m].mean()) for k in range(bins) if (b == k).any())


def _stamp(path: Path) -> list[int] | None:
    if not path.is_file():
        return None
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def _choices(dataset: str) -> dict[str, list[str]]:
    if dataset not in _CHOICES:
        frame = pd.read_parquet(RESOURCE / "datasets/test" / f"{dataset}.parquet")
        _CHOICES[dataset] = {str(r.problem_id): list(r.choices) for r in frame.itertuples()}
    return _CHOICES[dataset]


def metrics(y: np.ndarray, q: np.ndarray, q_valid: np.ndarray) -> dict:
    """Accuracy, Brier, ECE and AUROC of one bank: missing confidence scored 0."""
    c = np.where(q_valid, np.nan_to_num(q), 0.0)
    y = y.astype(float)
    auroc = float(roc_auc_score(y, c)) if 0 < y.mean() < 1 else float("nan")
    return {"acc": float(y.mean()), "brier": float(np.mean((c - y) ** 2)), "ece": float(ece(c, y)), "auroc": auroc,
            "n": int(len(y))}


def _build(bank: Path, dataset: str) -> dict:
    rows, lenient = [], []
    choices = _choices(dataset) if dataset in MC_SETS else None
    with (bank / "native.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            valid = bool(r.get("q_answer_parse_valid"))
            rows.append((str(r["problem_id"]), int(r["sample_id"]), float(r["q_answer"]) if valid else np.nan, valid,
                         bool(r.get("answer_parse_valid")), r.get("finish_reason") == "length",
                         bool(r.get("answer_correct"))))
            if choices is not None:
                gold = str(r.get("gold_answer")).strip().upper()
                lenient.append(lenient_letter(r, choices[str(r["problem_id"])]) == gold)
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    pick = lambda k, dtype: np.array([rows[i][k] for i in order], dtype=dtype)
    arrays = {"problem_id": pick(0, str), "sample_id": pick(1, np.int32), "q": pick(2, np.float32),
              "q_valid": pick(3, bool), "answer_parse_valid": pick(4, bool), "truncated": pick(5, bool),
              "y_strict": pick(6, bool)}
    if choices is not None:
        arrays["y_lenient"] = np.array([lenient[i] for i in order], dtype=bool)
    judge = bank / "llm_judge.jsonl"
    if judge.is_file():
        labels = {}
        with judge.open() as handle:
            for line in handle:
                r = json.loads(line)
                labels[(str(r["problem_id"]), int(r["sample_id"]))] = (bool(r["llm_correct_lenient"]), bool(r["llm_correct"]))
        keys = list(zip(arrays["problem_id"], arrays["sample_id"].tolist()))
        if all(k in labels for k in keys):
            arrays["y_llm"] = np.array([labels[k][0] for k in keys], dtype=bool)
            arrays["y_llm_strict"] = np.array([labels[k][1] for k in keys], dtype=bool)
    return arrays


def load(bank: Path | str, dataset: str | None = None, *, write: bool = True) -> dict | None:
    """The cached arrays of a complete bank (``None`` when the bank is incomplete); rebuilt when stale."""
    bank = Path(bank)
    if not (bank / "completion.json").is_file() or not (bank / "native.jsonl").is_file():
        return None
    dataset = dataset or bank.parent.name
    stamp = {"native": _stamp(bank / "native.jsonl"), "llm_judge": _stamp(bank / "llm_judge.jsonl")}
    cache, summary = bank / "scores.npz", bank / "summary.json"
    if cache.is_file() and summary.is_file():
        try:
            if json.loads(summary.read_text()).get("stamp") == stamp:
                with np.load(cache, allow_pickle=False) as data:
                    return {k: data[k] for k in data.files}
        except (OSError, ValueError, KeyError):
            pass
    arrays = _build(bank, dataset)
    if write:
        labels = [k for k in ("y_strict", "y_lenient", "y_llm", "y_llm_strict") if k in arrays]
        out = {"dataset": dataset, "stamp": stamp, "questions": int(len(np.unique(arrays["problem_id"]))),
               "metrics": {k[2:]: metrics(arrays[k], arrays["q"], arrays["q_valid"]) for k in labels},
               "confidence_parse_rate": float(arrays["q_valid"].mean()),
               "answer_parse_rate": float(arrays["answer_parse_valid"].mean()),
               "truncation_rate": float(arrays["truncated"].mean())}
        tmp = bank / "scores.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(cache)
        summary.write_text(json.dumps(out, indent=1) + "\n")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ARTIFACTS)
    parser.add_argument("--model", nargs="*", default=None, help="backbone directories (default: all)")
    args = parser.parse_args()
    models = args.model or [p.name for p in sorted(args.root.iterdir()) if (p / "pact").is_dir()]
    built = fresh = 0
    for model in models:
        for bank in sorted((args.root / model).glob("*/seed-*/evaluations/step-*/*/k*-seed-*")):
            if "aborted" in str(bank) or not (bank / "completion.json").is_file():
                continue
            before = (bank / "summary.json").stat().st_mtime_ns if (bank / "summary.json").is_file() else None
            load(bank)
            after = (bank / "summary.json").stat().st_mtime_ns if (bank / "summary.json").is_file() else None
            built += before != after
            fresh += before == after
    print(f"cache: {built} built, {fresh} already fresh")


if __name__ == "__main__":
    main()
