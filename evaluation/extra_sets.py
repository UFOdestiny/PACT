"""Evaluate finished runs on datasets their ``experiment.yaml`` snapshot does not list.

The snapshot of a finished run is never edited.  This reads it, adds the datasets to an in-memory copy of its
evaluation block, and runs training.run's own sampling and judging commands for them (SimpleQA also through
evaluation.llm_judge), so the new banks sit next to the others under ``evaluations/step-<max_steps>/<dataset>/`` with
the same seed, temperature, sample count and token budget.  Complete banks are skipped.  ``--seed-offset`` draws an
independent bank with another sampling seed (it lands in ``k<n>-seed-<offset + seed>``).

    python -m evaluation.extra_sets --runs artifacts/llama-3.1-8b-instruct/conftuner-pact/seed-41 --gpus 1
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

from evaluation.suite import ID, OOD
from training.experiment import (cell_path, checkpoint_complete, checkpoint_path, data_path, evaluation_complete,
                                 run_path)
from training.run import execute, generation_command, run_shards, scoring_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="run directories (<...>/seed-<n>)")
    parser.add_argument("--datasets", nargs="+", default=list(ID + OOD))
    parser.add_argument("--gpus", default="0", help="comma-separated devices for sampling")
    parser.add_argument("--seed-offset", type=int, default=None, help="independent resample with this evaluation seed offset")
    args = parser.parse_args()
    jobs = []
    for run in args.runs:
        config = yaml.safe_load((run / "experiment.yaml").read_text())
        seed = int(run.name.split("-")[-1])
        if not checkpoint_complete(checkpoint_path(config, seed), config["training"]["max_steps"],
                                   resumable=True, method=config["method"]):
            print(f"SKIP {run}: no complete checkpoint")
            continue
        extended = copy.deepcopy(config)
        if args.seed_offset is not None:
            extended["evaluation"]["seed_offset"] = args.seed_offset
        for dataset in args.datasets:
            extended["evaluation"]["datasets"].setdefault(dataset, len(pd.read_parquet(data_path(dataset))))
        e = extended["evaluation"]
        for dataset in args.datasets:
            problems = e["datasets"][dataset]
            if evaluation_complete(cell_path(extended, seed, dataset), problems=problems, samples=e["samples"],
                                   seed=e["seed_offset"] + seed, max_tokens=e["max_tokens"]):
                print(f"SKIP {run.parent.name}/{dataset}: complete")
                continue
            jobs.append((extended, seed, dataset, problems))
    sampling = [c for c in (generation_command(cfg, seed, d) for cfg, seed, d, _ in jobs) if c is not None]
    print(f"{len(jobs)} cells, {len(sampling)} to sample on GPUs {args.gpus}")
    run_shards(sampling, [{"CUDA_VISIBLE_DEVICES": g} for g in args.gpus.split(",")])
    run_shards([scoring_command(cfg, seed, d, n) for cfg, seed, d, n in jobs], [None] * min(len(jobs), 8))
    judged = sorted({str(run_path(cfg, seed)) for cfg, seed, d, _ in jobs if d == "simpleqa"})
    if judged:
        execute([sys.executable, "-m", "evaluation.llm_judge", "--datasets", "simpleqa", "--runs", *judged],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
    # every judged bank also gets its score cache (evaluation.score_cache: scores.npz + summary.json)
    from evaluation.score_cache import load
    for cfg, seed, d, _ in jobs:
        try:
            load(cell_path(cfg, seed, d), d)
        except Exception as error:
            print(f"WARNING score cache {d}: {error}", flush=True)


if __name__ == "__main__":
    main()
