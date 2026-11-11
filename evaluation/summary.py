"""Response-level confidence readouts and the four metrics on a native bank."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from training.registry import method_family
from evaluation import tasks
from evaluation.identity import cluster
from evaluation.metrics import ECE_BINS, METRICS, prepare, score

PRIMARY_READOUT = "consensus"
IDENTITY_MODE = "semantic"
# The channel each method family is scored on (base: the primary readout).
NATIVE_READOUTS = {"rlvr": "consensus", "rlcr": "verbal", "capo": "mean_token_logprob",
                   "dcpo": "verbal", "conftuner": "verbal", "pact": "verbal",
                   "credo": "verbal"}   # CREDO's two-token readout, stored as the bank's q_answer
VERBAL_METHODS = frozenset(m for m, r in NATIVE_READOUTS.items() if r == "verbal")


def hybrid_readout(consensus: np.ndarray, verbal: np.ndarray, k: int) -> np.ndarray:
    """Consensus with the verbal confidence as a within-level tie-break.

    The leave-one-out consensus takes only K discrete values, so responses at
    the same agreement level cannot be ranked against each other.  Shifting each
    value by ``(verbal - 0.5) / (2 (K - 1))`` orders responses inside a level by
    their verbal confidence while every level stays inside its own half-open
    band, so the ranking across levels and the calibration of the consensus are
    preserved up to that small offset.  A diagnostic readout, not a method.
    """
    if k <= 1:
        raise ValueError("the hybrid readout needs K > 1")
    values = consensus + (verbal - 0.5) / (2.0 * (k - 1))
    return np.clip(values, 0.0, 1.0)


def mean_token_logprob(raw):
    values = raw.get("token_logprobs")
    if not values or any(v is None for v in values):
        return None
    return float(np.mean(values))


def load(
    path: Path,
    *,
    expected_samples: int,
    expected_problems: int,
    dataset: str | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Rows of a bank with their leave-one-out consensus under the semantic answer identity."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if dataset is not None and raw.get("dataset") != dataset:
                raise ValueError(f"{path}:{line_number}: wrong dataset")
            if seed is not None and raw.get("seed") != seed:
                raise ValueError(f"{path}:{line_number}: wrong seed")
            if max_tokens is not None and raw.get("generation_max_tokens") != max_tokens:
                raise ValueError(f"{path}:{line_number}: wrong generation budget")
            try:
                row = {
                    "dataset": str(raw["dataset"]),
                    "method": str(raw.get("method", "")),
                    "problem_id": str(raw["problem_id"]),
                    "sample_id": int(raw["sample_id"]),
                    "parsed_answer": raw.get("parsed_answer"),
                    "answer_correct": bool(raw["answer_correct"]),
                    "judge_stage": str(raw.get("judge_stage", "")),
                    "verbal": float(raw["q_answer"]),
                    "verbal_parse_valid": bool(raw.get("q_answer_parse_valid", False)),
                    "completion_tokens": int(raw.get("completion_tokens", 0)),
                    "prompt_tokens": int(raw.get("prompt_tokens", 0)),
                    "mean_token_logprob": mean_token_logprob(raw),
                    "answer_parse_valid": bool(
                        raw.get("answer_parse_valid", raw.get("parsed_answer") is not None)
                    ),
                    "full_schema_compliant": bool(
                        raw.get(
                            "full_schema_compliant",
                            raw.get("format_compliant", False)
                            and raw.get("parsed_answer") is not None,
                        )
                    ),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: incomplete native row") from error
            if not 0.0 <= row["verbal"] <= 1.0:
                raise ValueError(f"{path}:{line_number}: invalid verbal confidence")
            groups[(row["dataset"], row["problem_id"])].append(row)
    if len(groups) != expected_problems:
        raise ValueError(f"{path}: expected {expected_problems} problems, found {len(groups)}")
    flattened: list[dict[str, Any]] = []
    problems = sorted(groups.items())
    for (dataset_name, problem_id), rows in problems:
        rows.sort(key=lambda row: row["sample_id"])
        if [row["sample_id"] for row in rows] != list(range(expected_samples)):
            raise ValueError(f"{path}: {dataset_name}/{problem_id} has incomplete sample IDs")
    jobs = [([r["parsed_answer"] for r in rows], [r["sample_id"] for r in rows], dataset_name)
            for (dataset_name, _), rows in problems]
    for ((dataset_name, problem_id), rows), identities in zip(problems, cluster_problems(jobs), strict=True):
        counts = Counter(identities)
        for row, label in zip(rows, identities, strict=True):
            row["problem_key"] = f"{dataset_name}\0{problem_id}"
            row["identity"] = label
            row["answer_count"] = counts[label]
            row["consensus"] = ((counts[label] - 1) / (expected_samples - 1)
                                if expected_samples > 1 else None)
            flattened.append(row)
    return flattened


def _cluster(job):
    return cluster(*job)


def cluster_problems(jobs: list[tuple[list, list, str]]) -> list[list[str]]:
    """``cluster`` for every problem, in order.  Problems are independent, so the symbolic math sets are
    spread over ``PACT_JUDGE_WORKERS`` processes (default 1: serial); each worker runs on its main
    thread, where math_verify's signal-based timeout works."""
    workers = min(int(os.environ.get("PACT_JUDGE_WORKERS", "1")), len(jobs))
    if workers <= 1 or jobs[0][2] in tasks.CHOICE_DATASETS | tasks.SHORT_ANSWER_DATASETS | tasks.CODE_DATASETS:
        return [_cluster(job) for job in jobs]

    def distinct(answers: list) -> int:
        seen: list = []
        for answer in answers:
            if answer not in seen:
                seen.append(answer)
        return len(seen)

    # Problems with the most distinct answers start first, so a slow symbolic problem is not left for last.
    order = sorted(range(len(jobs)), key=lambda i: -distinct(jobs[i][0]))
    labels: list = [None] * len(jobs)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, result in zip(order, pool.map(_cluster, [jobs[i] for i in order], chunksize=1)):
            labels[i] = result
    return labels


def metrics(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, float]:
    prepared_rows = [
        {"problem_id": row["problem_key"], "answer_correct": row["answer_correct"],
         "q_answer": float(probability)}
        for row, probability in zip(rows, probabilities, strict=True)
    ]
    problems = sorted({row["problem_key"] for row in rows})
    return score(prepare(prepared_rows, problems), np.ones(len(problems), dtype=np.int64))


def readout_values(rows, k):
    counts = np.asarray([r["answer_count"] for r in rows], dtype=float)
    verbal = np.asarray([r["verbal"] for r in rows], dtype=float)
    method = method_family(rows[0].get("method", "")) if rows else ""
    has_verbal = method in VERBAL_METHODS or not method
    values = {"consensus": (counts - 1) / (k - 1) if k > 1 else None,
              "verbal": verbal if has_verbal else None}
    values["hybrid"] = hybrid_readout(values["consensus"], verbal, k) if k > 1 and has_verbal else None
    logps = [r.get("mean_token_logprob") for r in rows]
    values["mean_token_logprob"] = (np.asarray(logps, dtype=float)
                                    if all(v is not None for v in logps) else None)
    return values


def score_readout(rows, values, *, name, rule="current"):
    labels = np.asarray([r["answer_correct"] for r in rows], dtype=bool)
    acc = float(labels.mean())
    if values is None:
        return {"acc": acc, "brier": None, "auroc": None, "ece": None}
    values = values.copy()
    if name == "mean_token_logprob":
        # Log likelihood is a ranking score, not a correctness probability.
        auc = float(roc_auc_score(labels, values)) if 0 < labels.sum() < len(labels) else None
        return {"acc": acc, "brier": None, "auroc": auc, "ece": None}
    if rule == "neutral":
        valid = np.asarray([r["answer_parse_valid"] for r in rows])
        values[~valid] = 0.5
    elif rule != "current":
        raise ValueError("unknown failure rule")
    return metrics(rows, values)


def generation_cost(rows, problems):
    prompts = {}
    for r in rows:
        prompts[r["problem_key"]] = r.get("prompt_tokens", 0)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in rows)
    return {"responses": len(rows), "total_prompt_tokens": sum(prompts.values()),
            "total_completion_tokens": completion_tokens,
            "completion_tokens_per_response": completion_tokens / len(rows),
            "tokens_per_problem": (sum(prompts.values()) + completion_tokens) / problems}


def group_composition(rows, k):
    """Share of problems whose K responses are all correct / all wrong / mixed.

    This is the evaluation-side view of the saturation diagnostic: a consensus
    readout has no ranking information inside a unanimous problem."""
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["problem_key"]].append(bool(r["answer_correct"]))
    counts = np.asarray([sum(v) for v in by_problem.values()], dtype=float)
    return {"unanimous_correct": float(np.mean(counts == k)),
            "unanimous_wrong": float(np.mean(counts == 0)),
            "mixed": float(np.mean((counts > 0) & (counts < k))),
            "pass_rate_histogram": np.bincount(counts.astype(int), minlength=k + 1).tolist()}


def analyze(path: Path, *, expected_samples: int, expected_problems: int, rows=None):
    if rows is None:
        rows = load(path, expected_samples=expected_samples, expected_problems=expected_problems)
    values = readout_values(rows, expected_samples)
    reports = {rule: {name: score_readout(rows, value, name=name, rule=rule)
                      for name, value in values.items()} for rule in ("current", "neutral")}
    stages = Counter(r["judge_stage"] for r in rows if r["judge_stage"])
    return {
        "path": str(path), "expected_samples": expected_samples,
        "expected_problems": expected_problems, "metrics": list(METRICS),
        "ece": {"bins": ECE_BINS, "strategy": "equal_width", "range": [0, 1]},
        "primary_readout": PRIMARY_READOUT, "identity": IDENTITY_MODE,
        "acc_definition": "sampled pass@1: mean correctness of all K responses per problem; no voting",
        "definitions": {
            "consensus": "(answer_count - 1) / (K - 1) under the stated answer identity",
            "verbal": "method-native verbal confidence; unavailable or missing confidence is 0",
            "mean_token_logprob": "mean raw-model log probability over completion tokens; Brier/ECE undefined",
            "hybrid": "consensus + (verbal - 0.5) / (2 (K - 1)): verbal confidence breaks ties inside a consensus level",
        },
        "native_readouts": NATIVE_READOUTS,
        "readouts": reports["current"], "readouts_by_rule": reports,
        "cost": generation_cost(rows, expected_problems),
        "readout_cost": {name: {"responses_required_per_score": expected_samples
                                 if name in ("consensus", "hybrid") else 1} for name in values},
        "format_compliance": {
            "answer_only_parse_rate": float(np.mean([r["answer_parse_valid"] for r in rows])),
            "full_schema_format_compliance_rate": float(np.mean([r["full_schema_compliant"] for r in rows])),
        },
        "judge_stages": dict(sorted(stages.items())),
        "composition": group_composition(rows, expected_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-problems", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.predictions, expected_samples=args.expected_samples,
                     expected_problems=args.expected_problems)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
