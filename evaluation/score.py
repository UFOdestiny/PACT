"""Audit a response bank and write four response-level metrics and costs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from evaluation.summary import analyze, load
from training.experiment import data_path


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def finalize(
    path: Path, *, dataset: str, problems: int, samples: int, seed: int, max_tokens: int
):
    native = path / "native.jsonl"
    rows = load(
        native,
        expected_samples=samples,
        expected_problems=problems,
        dataset=dataset,
        seed=seed,
        max_tokens=max_tokens,
    )
    source_ids = {
        str(x)
        for x in pq.read_table(data_path(dataset), columns=["problem_id"])[
            "problem_id"
        ].to_pylist()
    }
    if {row["problem_id"] for row in rows} != source_ids:
        raise ValueError(f"{native}: source dataset problem IDs differ")
    response = analyze(
        native, expected_samples=samples, expected_problems=problems, rows=rows
    )
    resources = path / "resources.json"
    if resources.is_file():
        response["cost"].update(json.loads(resources.read_text()))
    write_json(path / "response_readouts.json", response)
    stat = native.stat()
    write_json(
        path / "completion.json",
        {
            "native_bytes": stat.st_size,
            "native_mtime_ns": stat.st_mtime_ns,
            "rows": len(rows),
            "problems": problems,
            "samples": samples,
            "dataset": dataset,
            "seed": seed,
            "max_tokens": max_tokens,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--problems", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = vars(parser.parse_args())
    finalize(**args)


if __name__ == "__main__":
    main()
