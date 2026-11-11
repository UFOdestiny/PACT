"""Train and evaluate one experiment (``configs/<model>/<method>.yaml``), reusing completed stages.

    python -m training.run configs/llama-3.1-8b-instruct/conftuner-pact.yaml --seed 41 [--stage train|evaluate]

The launcher derives the generation batch and accumulation steps from the configured ``rollouts_per_update`` and the
world size, so every supported GPU count trains on the same responses per optimizer update.  Evaluation samples the
datasets on the same GPUs, one generation process per device, then judges the banks on CPU.  The resolved
experiment is snapshotted as ``experiment.yaml`` and every later stage must match it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from tools.model_protocol import closes_think_block, load_chat_tokenizer
from training.experiment import (
    RESOURCE,
    ROOT,
    cell_path,
    checkpoint_complete,
    checkpoint_path,
    data_path,
    evaluation_complete,
    load_config,
    resolve,
    run_path,
    scoring_marker,
)

DEFAULT_PORT = 29500


def validate_data_manifest() -> None:
    """Require the difficulty-annotated train/test manifest (tools/estimate_pass_rate.py, tools/build_manifest.py)."""
    provenance = RESOURCE / "datasets/provenance/manifest.json"
    try:
        manifest = json.loads(provenance.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "the experiment requires the difficulty-annotated manifest; "
            "run tools.estimate_pass_rate and tools.build_manifest first"
        ) from error
    expected = {
        "dataset_protocol": "mathstral_difficulty_stratified",
        "response_protocol": "screening",
        "sizes": {"train": 8000, "dev": 500, "test": 2000},
    }
    labeler = manifest.get("labeler", {})
    valid = (all(manifest.get(key) == value for key, value in expected.items())
             and labeler.get("name") == "mistralai/Mathstral-7B-v0.1"
             and labeler.get("samples") == 8
             and labeler.get("temperature") == 0.7
             and manifest.get("selection", {}).get("calibration_set") is None)
    if not valid:
        raise RuntimeError(f"{provenance} is not the K=8 / T=0.7 difficulty-annotated manifest with "
                           "train/dev/test = 8000/500/2000")


def visible_devices() -> list[str]:
    """The GPUs this process owns, named as ``CUDA_VISIBLE_DEVICES`` names them.

    The entries are passed through verbatim to the evaluation workers, because a child process reads
    ``CUDA_VISIBLE_DEVICES`` against the physical devices rather than against its parent's selection.
    """
    owned = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if owned:
        return owned
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    count = sum(line.startswith("GPU ") for line in listing.stdout.splitlines())
    return [str(index) for index in range(max(count, 1))]


def batch_settings(settings: dict, world_size: int) -> dict:
    """Split ``rollouts_per_update`` over the world so one update is always the same size; a world size that does
    not divide the update evenly is refused rather than rounded."""
    rollouts = int(settings.pop("rollouts_per_update"))
    micro = int(settings["per_device_train_batch_size"])
    group = int(settings["num_generations"])
    if rollouts % group:
        raise ValueError(f"{rollouts} responses per update do not divide into groups of {group}")
    if rollouts % (micro * world_size):
        feasible = [w for w in range(1, rollouts + 1) if not rollouts % (micro * w)]
        raise ValueError(
            f"{rollouts} responses per update do not divide over {world_size} x {micro}; "
            f"with this micro batch use one of {feasible[:8]} GPUs")
    settings["generation_batch_size"] = rollouts
    settings["gradient_accumulation_steps"] = rollouts // (micro * world_size)
    return settings


def execute(command, *, env=None, log=None):
    """Run ``command`` from the repository root, appending its output to ``log`` when given."""
    print(" ".join(map(str, command)), flush=True)
    if log:
        with log.open("a") as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def train(config, seed, *, dry_run=False):
    validate_data_manifest()
    validate_identity(config, seed, write=not dry_run)
    checkpoint = checkpoint_path(config, seed)
    settings = {k: v for k, v in config["training"].items() if k not in DEV_KEYS}
    if checkpoint_complete(checkpoint, settings["max_steps"], resumable=True, method=config["method"]):
        print(f"SKIP train {checkpoint}")
        return
    start = resolve(config["model"])
    if dry_run:
        print(f"TRAIN {config['method']} seed={seed} start={start} -> {checkpoint}")
        return
    if not (start / "config.json").is_file():
        raise FileNotFoundError(f"Missing model: {start}")
    devices = visible_devices()
    workspace = run_path(config, seed) / "training"
    workspace.mkdir(parents=True, exist_ok=True)
    if list((workspace / "checkpoints").glob("checkpoint-*")):
        raise RuntimeError(f"Previous staged checkpoints remain in {workspace}; resolve that interrupted run first")
    # Training writes the library's checkpoint names in its own staging directory; the completed checkpoint is then
    # renamed into the run layout.
    settings = batch_settings(settings, len(devices))
    settings.update(
        model_name_or_path=str(start),
        seed=seed,
        output_dir=str(workspace / "checkpoints"),
        run_name=f"{config['backbone']}-{config['method']}-s{seed}",
    )
    manifests = config.get("training_prompt_manifests", {})
    manifest = manifests.get(str(seed), manifests.get(seed))
    if manifest is not None:
        settings["data_train_file"] = manifest
        settings["shuffle_dataset"] = False
    settings["data_train_file"] = str(resolve(settings["data_train_file"]))
    think_closed = closes_think_block(load_chat_tokenizer(start))
    if think_closed:
        # TrlParser sets this env block in every training process before the reward functions are built.
        settings["env"] = {**settings.get("env", {}), "PACT_THINK_CLOSED": "1"}
    resolved = workspace / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(settings, sort_keys=False))
    # PYTORCH_CUDA_ALLOC_CONF is left alone: expandable segments are incompatible with colocated vLLM's
    # sleep-mode allocator (pytorch #147851).
    env = dict(
        os.environ,
        PACT_METHOD=config["method"],
        PACT_METHOD_OPTIONS=json.dumps(config.get("method_options", {})),
        PACT_THINK_CLOSED="1" if think_closed else "0",
        PACT_RETURN_VLLM_LOGPS="1",                 # the vendored trainer keeps vLLM's sampling log-probabilities
        PACT_DETERMINISTIC_GENERATION_SEEDS="1",    # per-request vLLM seeds derived from the run seed
        PACT_TRAIN_TRACE=str(workspace / "prompt_trace.jsonl"),
        WANDB_DISABLED="true",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=os.pathsep.join(map(str, [ROOT, ROOT / "third_party/trl", ROOT / "third_party/rlcr"])),
    )
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(len(devices)),
        "--main_process_port", str(config.get("port", DEFAULT_PORT)),
        "--config_file", str(ROOT / "configs/accelerate.yaml"),
        str(ROOT / "training/entrypoint.py"), "--config", str(resolved),
    ]
    execute(command, env=env, log=workspace / "train.log")
    produced = workspace / "checkpoints" / f"checkpoint-{settings['max_steps']}"
    weights_only = bool(settings.get("save_only_model"))
    if not checkpoint_complete(produced, settings["max_steps"], resumable=not weights_only, method=config["method"]):
        raise RuntimeError(f"Training did not produce a complete checkpoint: {produced}")
    if weights_only:
        # ``save_only_model`` never writes the optimizer shards; a finished run is never resumed, so the checkpoint
        # is marked the way tools/finalize_checkpoints marks one whose shards were removed.
        (produced / "finalized.json").write_text(json.dumps({
            "removed": None, "bytes": 0,
            "reason": f"save_only_model: optimizer shards never written; training finished at step {settings['max_steps']}",
            "when": time.strftime("%F %T")}, indent=2) + "\n")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted((workspace / "checkpoints").glob("checkpoint-*")):
        target = checkpoint.parent / path.name.replace("checkpoint-", "step-")
        if target.exists():
            raise FileExistsError(f"Existing checkpoint requires manual resolution: {target}")
        path.rename(target)


def generation_command(config, seed, dataset) -> list[str] | None:
    """The sampling command for one dataset, or ``None`` when its bank already exists."""
    e = config["evaluation"]
    path = cell_path(config, seed, dataset)
    path.mkdir(parents=True, exist_ok=True)
    native = path / "native.jsonl"
    if native.exists():
        return None
    return [
        sys.executable, "-m", "evaluation.generate",
        "--model", str(checkpoint_path(config, seed)), "--data", str(data_path(dataset)), "--output", str(native),
        "--method", config["method"], "--dataset", dataset,
        "--seed", str(e["seed_offset"] + seed), "--temperature", str(e["temperature"]),
        "--top-p", str(e["top_p"]), "--n", str(e["samples"]), "--max-tokens", str(e["max_tokens"]),
        "--tensor-parallel-size", "1",
    ]


def scoring_command(config, seed, dataset, problems) -> list[str]:
    """The judging command for one dataset's bank: CPU only, at low priority (it may share the node with the next
    job's sampling, which is bound by its main thread)."""
    e = config["evaluation"]
    return [
        "nice", "-n", "15", sys.executable, "-m", "evaluation.score", str(cell_path(config, seed, dataset)),
        "--dataset", dataset, "--problems", str(problems), "--samples", str(e["samples"]),
        "--seed", str(e["seed_offset"] + seed), "--max-tokens", str(e["max_tokens"]),
    ]


def run_shards(commands: list[list[str]], slots: list[dict | None]) -> None:
    """Run ``commands`` over ``slots`` workers, one command at a time per worker.  A slot is the environment overlay
    of its worker (a device for sampling, ``None`` for judging); every failure is collected and the first re-raised."""
    queue = list(commands)
    failures: list[BaseException] = []

    def worker(overlay: dict | None) -> None:
        while True:
            try:
                command = queue.pop(0)
            except IndexError:
                return
            try:
                execute(command, env=dict(os.environ, **overlay) if overlay else None)
            except BaseException as error:   # noqa: BLE001 - reported after every worker stops
                failures.append(error)
                return

    with ThreadPoolExecutor(max_workers=max(len(slots), 1)) as pool:
        list(pool.map(worker, slots))
    if failures:
        raise failures[0]


def evaluate(config, seed, *, dry_run=False):
    """Sample every dataset on the run's GPUs, LLM-judge SimpleQA on the first GPU, then judge the rest on CPU.

    Judging is a single-threaded CPU walk over a bank and is far slower than sampling for the symbolic datasets, so
    sampling takes one worker per GPU and judging one worker per dataset; the ``scoring`` marker tells a scheduler
    that the GPUs are free while the CPU judging runs.
    """
    validate_data_manifest()
    validate_identity(config, seed, write=not dry_run)
    marker = scoring_marker(config, seed)
    if not dry_run:
        marker.unlink(missing_ok=True)   # a retry samples again; it is not judging yet
    e = config["evaluation"]
    pending = []
    for dataset, problems in e["datasets"].items():
        path = cell_path(config, seed, dataset)
        if evaluation_complete(path, problems=problems, samples=e["samples"],
                               seed=e["seed_offset"] + seed, max_tokens=e["max_tokens"]):
            print(f"SKIP evaluate {path}")
            continue
        if dry_run:
            print(f"EVALUATE {path}")
            continue
        pending.append((dataset, problems))
    if dry_run:
        return
    devices = visible_devices()
    if pending:
        checkpoint = checkpoint_path(config, seed)
        if not checkpoint_complete(checkpoint, config["training"]["max_steps"], resumable=True,
                                   method=config["method"]):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        # The largest datasets first, so the last sampling wave is short whatever the device count is.
        pending.sort(key=lambda item: -item[1])
        sampling = [c for c in (generation_command(config, seed, d) for d, _ in pending) if c is not None]
        run_shards(sampling, [{"CUDA_VISIBLE_DEVICES": device} for device in devices])
        llm_judged = [(d, n) for d, n in pending if d == "simpleqa"]
        run_shards([scoring_command(config, seed, d, n) for d, n in llm_judged], [None])
        pending = [item for item in pending if item not in llm_judged]
    if "simpleqa" in e["datasets"]:
        execute([sys.executable, "-m", "evaluation.llm_judge", "--datasets", "simpleqa",
                 "--runs", str(run_path(config, seed))], env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]))
    if pending:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{os.getpid()}\n")
        try:
            run_shards([scoring_command(config, seed, d, n) for d, n in pending], [None] * len(pending))
        finally:
            marker.unlink(missing_ok=True)
    write_score_caches(config, seed)


def write_score_caches(config, seed):
    """Write each complete bank's score cache (evaluation.score_cache: scores.npz + summary.json)."""
    from evaluation.score_cache import load
    for dataset in config["evaluation"]["datasets"]:
        try:
            load(cell_path(config, seed, dataset), dataset)
        except Exception as error:   # the cache is a convenience; the bank itself is complete
            print(f"WARNING score cache {dataset}: {error}", flush=True)


# Development-set settings of earlier runs; no run evaluates a development set during training, so they are dropped.
DEV_KEYS = ("data_test_file", "eval_subset_size", "dev_diagnostic_steps", "dev_diagnostic_samples",
            "dev_diagnostic_seed_offset")
# Settings that never change the trained policy or a bank are not part of a run's identity: the seed list, whether the
# final checkpoint also stores the optimizer state, the development-set settings, and the list of evaluation datasets
# (each dataset's bank is its own directory, judged under the shared sampling settings).
IGNORED_TRAINING_KEYS = ("save_only_model", *DEV_KEYS)


def _identity(experiment: dict) -> dict:
    training = {k: v for k, v in experiment.get("training", {}).items() if k not in IGNORED_TRAINING_KEYS}
    evaluation = {k: v for k, v in experiment.get("evaluation", {}).items() if k != "datasets"}
    return {**{k: v for k, v in experiment.items() if k not in ("seeds", "port", "config_path")},
            "training": training, "evaluation": evaluation}


def validate_identity(config, seed, *, write=False):
    """Compare the resolved experiment with the run's ``experiment.yaml`` snapshot (written on first use)."""
    snapshot = run_path(config, seed) / "experiment.yaml"
    if snapshot.exists():
        if _identity(yaml.safe_load(snapshot.read_text())) != _identity(config):
            raise ValueError(f"Configuration differs from {snapshot}; a changed experiment needs a new method ID")
    elif write:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        identity = {k: v for k, v in config.items() if k not in ("port", "config_path")}
        snapshot.write_text(yaml.safe_dump(identity, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, help="Default: every seed in the YAML")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    config = load_config(args.config)
    config["port"] = args.port
    for seed in [args.seed] if args.seed is not None else config["seeds"]:
        if seed not in config["seeds"]:
            parser.error(f"Seed {seed} is not declared in the experiment YAML")
        if args.dry_run:
            if args.stage != "evaluate":
                train(config, seed, dry_run=True)
            if args.stage != "train":
                evaluate(config, seed, dry_run=True)
            continue
        run = run_path(config, seed)
        run.mkdir(parents=True, exist_ok=True)
        with (run / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.stage != "evaluate":
                train(config, seed)
            if args.stage != "train":
                evaluate(config, seed)


if __name__ == "__main__":
    main()
