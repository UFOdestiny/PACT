"""Durable JSON metric history of a training run (``<run>/training/metrics.jsonl``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import TrainerCallback


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


class MetricsHistoryCallback(TrainerCallback):
    """Append every log event and record the resolved trainer config in each checkpoint."""

    @staticmethod
    def _run_dir(args) -> Path:
        return Path(args.output_dir).resolve().parent

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        path = self._run_dir(args) / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _json_safe({"step": state.global_step, **logs})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if any(
            field in record
            for field in ("rewards/accuracy_reward", "reward", "rewards/format_reward")
        ):
            from training.training_curve import render

            render(path, self._run_dir(args).parent / "training.png")

    def on_step_end(self, args, state, control, **kwargs):
        """Make the non-periodic final optimizer step a recoverable checkpoint."""
        if state.max_steps > 0 and state.global_step == state.max_steps:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        config = _json_safe(
            {
                "trainer": args.to_dict(),
                "checkpoint_step": state.global_step,
            }
        )
        (checkpoint / "run_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
