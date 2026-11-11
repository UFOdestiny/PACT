"""Remove the DeepSpeed optimizer shards of runs whose training has finished.

The protocol fixes the number of updates and nothing ever resumes a run, so once
the final weights are on disk the optimizer state is dead weight -- and it
is by far the largest artifact (about 100 GB per 7-8B checkpoint against 15 GB
of weights).  This removes ``checkpoints/step-<max_steps>/global_step<max_steps>/``
(the step is read from each run's configuration, ``configs/<model>/<run>.yaml``) and writes
``finalized.json`` in its place, which ``training.experiment.checkpoint_complete``
accepts instead of the resumable files, so the run keeps counting as complete.

Evaluation reads only the HuggingFace weights, so a run that is still
generating its banks can be finalized too.  The only wait is for the
checkpoint to be fully written: a directory whose newest file is younger than
``--settle`` seconds is left alone.

    python -m tools.finalize_checkpoints              # report what would go
    python -m tools.finalize_checkpoints --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from training.experiment import ROOT, checkpoint_complete, load_config, run_path


def newest_mtime(path: Path) -> float:
    return max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()), default=0.0)


def directory_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete; without it, only report")
    parser.add_argument("--step", type=int, help="override the configured max_steps")
    parser.add_argument("--settle", type=int, default=120, help="seconds a checkpoint must be untouched")
    args = parser.parse_args()

    freed = 0
    for experiment in sorted((ROOT / "configs").glob("*/*.yaml")):
        config = load_config(experiment)
        step = args.step or int(config["training"]["max_steps"])
        for seed in config["seeds"]:
            run = run_path(config, seed)
            shards = run / "checkpoints" / f"step-{step}" / f"global_step{step}"
            if not shards.is_dir():
                continue
            label = f"{config['backbone']}/{config['method']}/seed-{seed}"
            if not checkpoint_complete(shards.parent, step, method=config["method"]):
                print(f"SKIP {label}: step-{step} weights incomplete")
                continue
            age = time.time() - newest_mtime(shards)
            if age < args.settle:
                print(f"SKIP {label}: checkpoint written {age:.0f}s ago, still settling")
                continue
            size = directory_bytes(shards)
            freed += size
            print(f"{'FINALIZE' if args.apply else 'WOULD FINALIZE'} {label}: {size / 2**30:.1f} GiB")
            if args.apply:
                (shards.parent / "finalized.json").write_text(
                    json.dumps(
                        {
                            "removed": f"global_step{step}",
                            "bytes": size,
                            "reason": f"training finished at step {step}; the protocol never resumes",
                            "when": time.strftime("%F %T"),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                shutil.rmtree(shards)
    print(f"{'freed' if args.apply else 'would free'} {freed / 2**40:.2f} TiB")


if __name__ == "__main__":
    main()
