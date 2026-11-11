"""Build the one shared train/dev/test dataset from Mathstral difficulty labels.

The labeler measures every cleaned pool problem with K=8 stochastic samples at
T=0.7.  No difficulty bin is discarded.  Dev (500) and test (2,000) preserve
the pool difficulty distribution; train (8,000) is then balanced as evenly as
the remaining 0/8 ... 8/8 bins allow.  Split membership is fixed by one seed.
Training seeds only permute the same 8,000 training rows.

Outputs:

* ``train/prompts-seed-<seed>.parquet`` for each training seed,
* ``dev/pool-dev.parquet``, reserved (no experiment reads it),
* ``test/pool_heldout.parquet`` for final in-distribution evaluation only,
* ``provenance/manifest.json`` with exact selection rules and histograms.

There is deliberately no calibration split and no post-hoc calibration fit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training.experiment import ROOT, RESOURCE
from tools.prepare_train_pool import digitless, normalise, word_ngrams

DATASETS = RESOURCE / "datasets"
LABELER = "mistralai/Mathstral-7B-v0.1"
LABELER_BACKBONE = "mathstral-7b-v0.1"


def relative(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def attach_labels(pool: list[dict], rates: list[dict]) -> tuple[list[dict], int]:
    """Join one complete label row to every pool row and expose audit fields."""
    by_id = {str(row["problem_id"]): row for row in rates}
    if len(by_id) != len(rates):
        raise ValueError("pass-rate table contains duplicate problem_id values")
    pool_ids = [str(row["problem_id"]) for row in pool]
    if len(set(pool_ids)) != len(pool_ids):
        raise ValueError("pool contains duplicate problem_id values")
    if set(pool_ids) != set(by_id):
        missing = len(set(pool_ids) - set(by_id))
        extra = len(set(by_id) - set(pool_ids))
        raise ValueError(f"pass-rate coverage mismatch: missing={missing}, extra={extra}")
    sample_counts = {int(row["samples"]) for row in rates}
    if len(sample_counts) != 1:
        raise ValueError(f"inconsistent sample counts in pass-rate table: {sample_counts}")
    samples = sample_counts.pop()
    labelled = []
    for row in pool:
        label = by_id[str(row["problem_id"])]
        correct = int(label["correct"])
        rate = float(label["pass_rate"])
        if not 0 <= correct <= samples or not math.isclose(rate, correct / samples, abs_tol=1e-12):
            raise ValueError(f"invalid pass-rate label for {row['problem_id']}")
        labelled.append({**row, "pass_rate": rate, "labeler_correct": correct,
                         "labeler_samples": samples})
    return labelled, samples


def proportional_quotas(capacities: dict[int, int], total: int) -> dict[int, int]:
    """Largest-remainder allocation proportional to bin capacity."""
    available = sum(capacities.values())
    if total < 0 or total > available:
        raise ValueError(f"cannot draw {total} rows from capacity {available}")
    if total == 0:
        return {key: 0 for key in capacities}
    exact = {key: total * value / available for key, value in capacities.items()}
    quotas = {key: min(value, math.floor(exact[key])) for key, value in capacities.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(capacities, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    for key in priority:
        if remainder == 0:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError("largest-remainder allocation failed")
    return quotas


def balanced_quotas(capacities: dict[int, int], total: int, seed: int) -> dict[int, int]:
    """Max-min allocate training rows across difficulty bins."""
    if total < 0 or total > sum(capacities.values()):
        raise ValueError(f"cannot draw {total} rows from capacity {sum(capacities.values())}")
    quotas = {key: 0 for key in capacities}
    priority = sorted(capacities)
    random.Random(seed).shuffle(priority)
    rank = {key: index for index, key in enumerate(priority)}
    for _ in range(total):
        eligible = [key for key in capacities if quotas[key] < capacities[key]]
        key = min(eligible, key=lambda item: (quotas[item], rank[item]))
        quotas[key] += 1
    return quotas


def draw(groups: dict[int, list[dict]], quotas: dict[int, int]) -> list[dict]:
    selected = []
    for key in sorted(groups):
        count = quotas.get(key, 0)
        if count > len(groups[key]):
            raise ValueError(f"bin {key} requests {count} rows from {len(groups[key])}")
        selected.extend(groups[key][:count])
        del groups[key][:count]
    return selected


def deduplicate_templates(rows: list[dict], seed: int) -> tuple[list[dict], int]:
    """Keep one seeded representative of problems differing only in numbers."""
    order = list(rows)
    random.Random(seed).shuffle(order)
    kept, seen = [], set()
    for row in order:
        key = digitless(row["problem"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)


def quarantine_overlaps(groups: dict[int, list[dict]], references: list[dict]) -> dict[str, int]:
    """Remove candidates that resemble a frozen held-out split."""
    exact = {normalise(row["problem"]) for row in references}
    without_digits = {digitless(row["problem"]) for row in references}
    grams = set().union(*(word_ngrams(row["problem"]) for row in references))
    counts = {"exact": 0, "digitless": 0, "ngram": 0}
    for key, rows in groups.items():
        kept = []
        for row in rows:
            problem = row["problem"]
            if normalise(problem) in exact:
                counts["exact"] += 1
            elif digitless(problem) in without_digits:
                counts["digitless"] += 1
            elif word_ngrams(problem) & grams:
                counts["ngram"] += 1
            else:
                kept.append(row)
        groups[key] = kept
    return counts


def split(rows: list[dict], *, train: int, test: int, dev: int, seed: int
          ) -> tuple[list[dict], list[dict], list[dict], dict[str, dict[int, int]], dict]:
    """Reserve isolated proportional test/dev sets, then max-min balance train."""
    if len(rows) < train + test + dev:
        raise ValueError(f"only {len(rows)} labelled problems for {train} + {test} + {dev}")
    unique, template_duplicates = deduplicate_templates(rows, seed)
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in sorted(unique, key=lambda item: str(item["problem_id"])):
        groups[int(row["labeler_correct"])].append(row)
    for key, group in groups.items():
        random.Random(seed * 1009 + key).shuffle(group)
    capacities = {key: len(group) for key, group in groups.items()}
    test_quotas = proportional_quotas(capacities, test)
    test_rows = draw(groups, test_quotas)
    after_test_quarantine = quarantine_overlaps(groups, test_rows)
    after_test = {key: len(group) for key, group in groups.items()}
    dev_quotas = proportional_quotas(after_test, dev)
    dev_rows = draw(groups, dev_quotas)
    after_dev_quarantine = quarantine_overlaps(groups, dev_rows)
    after_dev = {key: len(group) for key, group in groups.items()}
    train_quotas = balanced_quotas(after_dev, train, seed)
    train_rows = draw(groups, train_quotas)
    random.Random(seed + 1).shuffle(test_rows)
    random.Random(seed + 2).shuffle(dev_rows)
    return train_rows, test_rows, dev_rows, {
        "train": train_quotas, "test": test_quotas, "dev": dev_quotas,
    }, {
        "template_deduplicated": template_duplicates,
        "selection_pool_size": len(unique),
        "quarantined_after_test": after_test_quarantine,
        "quarantined_after_dev": after_dev_quarantine,
        "remaining_after_selection": sum(len(group) for group in groups.values()),
    }


def histogram(rows: list[dict], samples: int) -> list[int]:
    counts = [0] * (samples + 1)
    for row in rows:
        counts[int(row["labeler_correct"])] += 1
    return counts


def dataset_row(row: dict, source: str | None = None) -> dict:
    return {
        "problem_id": str(row["problem_id"]),
        "problem": row["problem"],
        "answer": row["answer"],
        "origin": row["origin"],
        "source": source if source is not None else row["source"],
        "pass_rate": float(row["pass_rate"]),
        "labeler_correct": int(row["labeler_correct"]),
        "labeler_samples": int(row["labeler_samples"]),
    }


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def validate_labeler_report(pass_rates: Path, samples: int) -> dict:
    report_path = DATASETS / "provenance" / f"{pass_rates.stem}.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing labeler provenance: {report_path}")
    report = json.loads(report_path.read_text())
    expected = {
        "backbone": LABELER_BACKBONE,
        "samples": samples,
        "temperature": 0.7,
        "response_protocol": "screening",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"labeler provenance {key}={report.get(key)!r}; expected {value!r}")
    model = str(report.get("model", "")).rstrip("/")
    if not (model.endswith("Mathstral-7B-v0.1") or model == LABELER):
        raise ValueError(f"unexpected labeler model: {model!r}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DATASETS / "train/pool.parquet")
    parser.add_argument("--pass-rates", type=Path,
                        default=DATASETS / "train/passrate-mathstral-7b-v0.1-screening.parquet")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43, help="fixed membership seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43],
                        help="training seeds; these change row order, not membership")
    args = parser.parse_args()

    pool_path = args.pool if args.pool.is_absolute() else ROOT / args.pool
    pass_rates_path = args.pass_rates if args.pass_rates.is_absolute() else ROOT / args.pass_rates
    pool = pq.read_table(pool_path).to_pylist()
    rates = pq.read_table(pass_rates_path).to_pylist()
    labelled, samples = attach_labels(pool, rates)
    labeler_report = validate_labeler_report(pass_rates_path, samples)
    train, test, dev, quotas, isolation = split(
        labelled, train=args.train, test=args.test, dev=args.dev, seed=args.seed
    )

    outputs = {}
    for seed in args.seeds:
        order = list(train)
        random.Random(seed).shuffle(order)
        path = DATASETS / "train" / f"prompts-seed-{seed}.parquet"
        write(path, [dataset_row(row) for row in order])
        outputs[f"train_seed_{seed}"] = relative(path)
    test_path = DATASETS / "test/pool_heldout.parquet"
    dev_path = DATASETS / "dev/pool-dev.parquet"
    write(test_path, [dataset_row(row, "pool_heldout") for row in test])
    write(dev_path, [dataset_row(row, "pool_dev") for row in dev])
    outputs.update({"test": relative(test_path), "dev": relative(dev_path)})

    report = {
        "dataset_protocol": "mathstral_difficulty_stratified",
        "response_protocol": "screening",
        "pool": relative(pool_path),
        "pass_rates": relative(pass_rates_path),
        "labeler": {
            "name": LABELER,
            "backbone": LABELER_BACKBONE,
            "samples": samples,
            "temperature": 0.7,
            "system_prompt": labeler_report["system_prompt"],
            "max_tokens": labeler_report["max_tokens"],
        },
        "selection": {
            "difficulty_bins_kept": list(range(samples + 1)),
            "internal_deduplication": "one seeded representative per digit-stripped problem template",
            "cross_split_isolation": "exact, digit-stripped, and shared 13-word-window quarantine",
            "train": "max-min balanced across labeler_correct bins after isolated dev/test reservation",
            "dev": "proportional to the template-deduplicated pool; reserved",
            "test": "proportional to the template-deduplicated pool; final evaluation only",
            "calibration_set": None,
        },
        "pool_size": len(labelled),
        "isolation": isolation,
        "split_seed": args.seed,
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "correct_histogram": {
            "pool": histogram(labelled, samples),
            "train": histogram(train, samples),
            "dev": histogram(dev, samples),
            "test": histogram(test, samples),
        },
        "quotas": {name: [values.get(key, 0) for key in range(samples + 1)]
                   for name, values in quotas.items()},
        "outputs": outputs,
    }
    manifest_path = DATASETS / "provenance/manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
