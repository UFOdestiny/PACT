"""Per-step efficiency accounting for a training run.

Every optimizer step logs, through the trainer's metric dictionary and hence
``metrics.jsonl``:

* ``time/generation_s``  -- wall time of rollout generation plus reward scoring
  for the generation batch that fed this step (colocated vLLM, so this is the
  sampling cost the policy pays);
* ``time/optimizer_s``   -- wall time of the forward/backward/optimizer work;
* ``time/step_s``        -- everything between two optimizer steps;
* ``time/cumulative_s``  -- wall time since the first step, so a step budget can
  be converted to GPU-hours;
* ``tokens/completion``  -- completion tokens generated for the step,
  ``tokens/cumulative_completion`` their running total,
  ``tokens/completion_per_s`` the generation throughput;
* ``memory/peak_allocated_gb`` and ``memory/peak_reserved_gb`` -- CUDA peaks
  since the previous step.

At the end of training ``efficiency.json`` next to ``metrics.jsonl`` holds the
totals (the cost table).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


class StepTimer:
    """Accumulates phase timings between optimizer steps."""

    def __init__(self) -> None:
        self.reset()
        self.train_started: float | None = None
        self.cumulative_completion_tokens = 0
        self.cumulative_prompt_tokens = 0

    def reset(self) -> None:
        self.generation_s = 0.0
        self.optimizer_s = 0.0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.last_step_end = time.perf_counter()

    def add_generation(self, seconds: float, completion_tokens: int, prompt_tokens: int) -> None:
        self.generation_s += seconds
        self.completion_tokens += completion_tokens
        self.prompt_tokens += prompt_tokens
        self.cumulative_completion_tokens += completion_tokens
        self.cumulative_prompt_tokens += prompt_tokens

    def add_optimizer(self, seconds: float) -> None:
        self.optimizer_s += seconds

    def step_metrics(self) -> dict[str, float]:
        now = time.perf_counter()
        if self.train_started is None:
            self.train_started = self.last_step_end
        step_s = now - self.last_step_end
        metrics = {
            "time/generation_s": self.generation_s,
            "time/optimizer_s": self.optimizer_s,
            "time/step_s": step_s,
            "time/cumulative_s": now - self.train_started,
            "tokens/completion": float(self.completion_tokens),
            "tokens/prompt": float(self.prompt_tokens),
            "tokens/cumulative_completion": float(self.cumulative_completion_tokens),
            "tokens/completion_per_s": self.completion_tokens / self.generation_s if self.generation_s > 0 else 0.0,
        }
        if torch.cuda.is_available():
            metrics["memory/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            metrics["memory/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            torch.cuda.reset_peak_memory_stats()
        self.reset()
        return metrics


class EfficiencySummaryCallback(TrainerCallback):
    """Write ``efficiency.json`` with run totals when training ends."""

    def __init__(self, timer: StepTimer, metrics_path: Path) -> None:
        self.timer = timer
        self.metrics_path = metrics_path

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        rows = []
        if self.metrics_path.is_file():
            with self.metrics_path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        timed = [r for r in rows if "time/step_s" in r]
        summary = {
            "steps": int(state.global_step),
            "wall_seconds": sum(r["time/step_s"] for r in timed),
            "generation_seconds": sum(r["time/generation_s"] for r in timed),
            "optimizer_seconds": sum(r["time/optimizer_s"] for r in timed),
            "completion_tokens": self.timer.cumulative_completion_tokens,
            "prompt_tokens": self.timer.cumulative_prompt_tokens,
            "peak_allocated_gb": max((r.get("memory/peak_allocated_gb", 0.0) for r in timed), default=None),
            "gpus": int(getattr(args, "world_size", 1) or 1),
        }
        if summary["wall_seconds"] > 0:
            summary["gpu_hours"] = summary["wall_seconds"] * summary["gpus"] / 3600
            summary["completion_tokens_per_gpu_second"] = (
                summary["completion_tokens"] / (summary["wall_seconds"] * summary["gpus"]))
        (self.metrics_path.parent / "efficiency.json").write_text(json.dumps(summary, indent=2) + "\n")
