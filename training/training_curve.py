"""Render the live per-step training curve of a run from its ``metrics.jsonl`` log.

``training/trainer_metrics.py`` redraws ``<run>/training.png`` after every logged update.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


PANELS = (
    ("rewards/accuracy_reward", "On-policy rollout accuracy", "Accuracy", (0.0, 1.0)),
    ("reward", "Mean training reward", "Reward", None),
    ("rewards/format_reward", "Format compliance", "Compliance", (0.0, 1.0)),
)
PACT_COLOR = "#C8102E"     # the paper's colour for the PACT family
OTHER_COLOR = "#41464D"


def load_points(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [
        row
        for row in rows
        if "step" in row and any(field in row for field, *_ in PANELS)
    ]


def rolling_mean(values: list[float], window: int = 20) -> list[float]:
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def render(metrics: Path, output: Path) -> None:
    rows = load_points(metrics)
    if not rows:
        return

    matplotlib_cache = Path(tempfile.gettempdir()) / f"pact-matplotlib-{os.getuid()}"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.registry import method_family

    run = metrics.parent.parent                 # <model>/<method>/seed-<n>/training/metrics.jsonl
    method = run.parent.name
    backbone = run.parent.parent.name
    last_step = max(int(row["step"]) for row in rows)
    color = PACT_COLOR if method_family(method) == "pact" else OTHER_COLOR
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#41464D", "axes.linewidth": .7,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "text.usetex": False}):
        figure, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
        for axis, (field, title, ylabel, limits) in zip(axes, PANELS, strict=True):
            selected = [(int(row["step"]), float(row[field])) for row in rows if field in row]
            steps = [step for step, _ in selected]
            values = [value for _, value in selected]
            if not values:
                axis.text(.5, .5, "Not logged", transform=axis.transAxes, ha="center")
                continue
            axis.plot(steps, values, color=color, linewidth=.65, alpha=.22, label="Per update")
            axis.plot(steps, rolling_mean(values), color=color, linewidth=1.8,
                      label="20-update trailing mean")
            axis.scatter(steps[-1], rolling_mean(values)[-1], s=15, color=color, zorder=5)
            axis.set_title(title, loc="left", fontsize=10, pad=5)
            axis.set_ylabel(ylabel)
            if limits is not None:
                axis.set_ylim(*limits)
                axis.set_yticks([0, .25, .5, .75, 1])
            axis.grid(axis="y", color="#E5E7EB", linewidth=.6)
        axes[0].legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
        axes[-1].set_xlabel("Optimizer update")
        axes[-1].set_xlim(0, max(125, last_step))
        figure.suptitle(f"{backbone}  /  {method}\n{run.name} · recorded updates: {last_step}",
                       fontsize=12, y=.99)
        figure.tight_layout(rect=(0, 0, 1, .95), h_pad=1.1)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics.parent.parent / "training.png"
    render(args.metrics, output)
    print(output)


if __name__ == "__main__":
    main()
