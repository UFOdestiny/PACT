"""Experiment configuration (``configs/<model>/<method>.yaml``), local paths, and completeness checks.

A run's YAML names its model, method and evaluation and overrides the shared ``configs/training.yaml``;
``load_config`` resolves it into the experiment that ``training/run.py`` snapshots as
``<artifacts>/<model>/<method>/seed-<n>/experiment.yaml``.

Local paths come from ``.env`` (see ``.env.example``) or the environment: ``PACT_RESOURCE`` holds models and datasets
(default ``resource/``), ``PACT_ARTIFACTS`` the run directories (default ``artifacts/``).  Paths in the configs that
start with ``resource/`` or ``artifacts/`` are resolved against them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from tools.response_protocol import method_name, response_schema
from training import registry

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    """Read ``KEY=VALUE`` lines of a ``.env`` file into the environment; variables already set win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            value = os.path.expanduser(value)
            if value.startswith("./"):
                value = str(ROOT / value[2:])
            os.environ.setdefault(key.strip(), value)


load_env()
RESOURCE = Path(os.environ.get("PACT_RESOURCE") or ROOT / "resource")
ARTIFACTS = Path(os.environ.get("PACT_ARTIFACTS") or ROOT / "artifacts")


def resolve(path: str | Path) -> Path:
    """A path as written in a config: ``resource/...`` and ``artifacts/...`` under ``PACT_RESOURCE`` /
    ``PACT_ARTIFACTS``, any other relative path under the repository."""
    path = Path(path)
    if path.is_absolute():
        return path
    head, *rest = path.parts
    if head == "resource":
        return RESOURCE.joinpath(*rest)
    if head == "artifacts":
        return ARTIFACTS.joinpath(*rest)
    return ROOT / path


def load_config(path: Path) -> dict:
    """The resolved experiment: shared training settings, run overrides, and the method's response contract and
    reward specification."""
    path = resolve(path)
    config = yaml.safe_load(path.read_text())
    training = yaml.safe_load(resolve(config["training_config"]).read_text())
    training.update(config.get("training", {}))
    profile = method_name(config["method"])
    training["sys_prompt_name"] = profile
    training["format_pattern"] = response_schema(profile)
    if config["method"] != "base":
        funcs, weights = registry.reward_specification(config["method"])
        training["reward_funcs"] = funcs
        training["reward_weights"] = weights
    config["training"] = training
    config["evaluation"] = dict(config["evaluation"])
    config["evaluation"]["response_protocol"] = profile
    config["config_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    for field in ("backbone", "method"):
        if not config.get(field):
            raise ValueError(f"Missing {field}: {path}")
    return config


def run_path(config: dict, seed: int) -> Path:
    return ARTIFACTS / config["backbone"] / config["method"] / f"seed-{seed}"


def scoring_marker(config: dict, seed: int) -> Path:
    """Marker written while a run's banks are judged on the CPU (its GPUs are free by then); removed when the run
    ends and cleared on entry, so a restarted run never claims to be judging while it samples."""
    return run_path(config, seed) / "scoring"


def checkpoint_path(config: dict, seed: int) -> Path:
    if config["method"] == "base":
        # The untrained model evaluated under the protocol: no training stage; the "checkpoint" is the model itself.
        return resolve(config["model"])
    return run_path(config, seed) / "checkpoints" / f"step-{config['training']['max_steps']}"


def cell_path(config: dict, seed: int, dataset: str) -> Path:
    e = config["evaluation"]
    return (run_path(config, seed) / "evaluations" / f"step-{config['training']['max_steps']}"
            / dataset / f"k{e['samples']}-seed-{e['seed_offset'] + seed}")


def data_path(dataset: str) -> Path:
    """Frozen evaluation split of one dataset."""
    return RESOURCE / "datasets/test" / f"{dataset}.parquet"


def checkpoint_complete(path: Path, step: int, *, resumable: bool = False, method: str | None = None) -> bool:
    """Whether ``path`` holds the weights of optimizer step ``step``; with ``resumable`` also the DeepSpeed
    resume state, unless ``finalized.json`` records that the optimizer shards were removed or never written."""
    if method == "base":
        return (path / "config.json").is_file()
    try:
        if not (path / "config.json").is_file():
            return False
        index = path / "model.safetensors.index.json"
        files = (set(json.loads(index.read_text())["weight_map"].values()) if index.exists()
                 else {"model.safetensors"})
        if not files or not all((path / name).is_file() and (path / name).stat().st_size > 0 for name in files):
            return False
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != step:
            return False
        if resumable and not (path / "finalized.json").is_file():
            if not all((path / name).is_file() for name in ("scheduler.pt", "latest")):
                return False
            # One process writes ``rng_state.pth``; a multi-process cell writes one
            # ``rng_state_<rank>.pth`` per rank, so either shape counts as resumable.
            if not ((path / "rng_state.pth").is_file() or list(path.glob("rng_state_*.pth"))):
                return False
            if not list((path / f"global_step{step}").glob("*_optim_states.pt")):
                return False
            if not list((path / f"global_step{step}").glob("*_model_states.pt")):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def evaluation_complete(path: Path, *, problems: int, samples: int, seed: int, max_tokens: int) -> bool:
    try:
        native = path / "native.jsonl"
        config = json.loads((path / "run_config.json").read_text())
        response = json.loads((path / "response_readouts.json").read_text())
        completion = json.loads((path / "completion.json").read_text())
        stat = native.stat()
        return (
            config["seed"] == seed
            and config["n"] == samples
            and config["max_tokens"] == max_tokens
            and response["metrics"] == ["acc", "brier", "auroc", "ece"]
            and response["primary_readout"] == "consensus"
            and response["expected_samples"] == samples
            and response["expected_problems"] == problems
            and completion["native_bytes"] == stat.st_size
            and completion["native_mtime_ns"] == stat.st_mtime_ns
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
