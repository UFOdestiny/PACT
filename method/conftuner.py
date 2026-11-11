"""ConfTuner's credit (Li et al., NeurIPS 2025), the confidence objective that ConfTuner+PACT extends.

The response ends with ``Confidence: <N>%``.  The reasoning tokens receive the shared group-centred correctness
advantage; every token of the confidence block gets zero policy-gradient advantage, so the stated number is trained
only by the expected (tokenized) Brier score over the percentage grid ``sum_i q_i (y - i/100)^2``, where ``q`` is the
policy's distribution over the 101 percentages at the number's position (``training/confidence_grid.py``, switched on
by the ``exact_confidence`` option).  The Brier term is summed over responses and divided by the micro-batch's
completion tokens, the scale of the token-level policy loss, and its gradient flows through the full response.

ConfTuner itself is the ``conftuner`` method: ``method/pact.py`` with the peer tests off.
"""

from __future__ import annotations

import numpy as np

from training.grpo import grpo_advantages, match_rms

REWARD_FUNCS = ["format", "accuracy"]
REWARD_WEIGHTS = [0.5, 0.5]
BLOCK_TAG = "Confidence:"


def conftuner_advantages(rewards: np.ndarray, labels: np.ndarray, block_start: np.ndarray,
                         completion_lengths: np.ndarray, *, completion_width: int, group_width: int,
                         scale_match: bool = True, formats: np.ndarray | None = None,
                         trained: np.ndarray | None = None, reduce_sum=None) -> tuple[np.ndarray, dict[str, float]]:
    """Reasoning tokens get the shared group-centred reward; the confidence block gets zero."""
    rewards = np.asarray(rewards, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=bool)
    starts = np.asarray(block_start, dtype=np.int64)
    lengths = np.asarray(completion_lengths, dtype=np.int64)
    n = len(outcomes)
    if not (rewards.shape == starts.shape == lengths.shape == (n,)):
        raise ValueError("ConfTuner inputs must be aligned one-dimensional vectors")
    if group_width <= 1 or n % group_width:
        raise ValueError("inputs must contain complete rollout groups")
    if completion_width < int(lengths.max(initial=0)):
        raise ValueError("completion_width is smaller than a completion")
    reasoning = rewards.reshape(-1, group_width)
    reasoning = (reasoning - reasoning.mean(axis=1, keepdims=True)).reshape(-1)
    positions = np.arange(completion_width)[None, :]
    valid = positions < lengths[:, None]
    boundary = np.where(starts >= 0, starts, lengths)          # no tag: the whole response is reasoning
    in_confidence = positions >= boundary[:, None]
    advantages = np.where(in_confidence, 0.0, reasoning[:, None]) * valid
    factor = 1.0
    if scale_match:
        kept = np.ones(n, dtype=bool) if trained is None else np.asarray(trained, dtype=bool)
        reference = grpo_advantages(np.ones(n) if formats is None else formats, outcomes, group_width=group_width)
        advantages, factor = match_rms(advantages, reference, mask=valid & kept[:, None], reduce_sum=reduce_sum)
    diagnostics = {
        "conftuner_tag_missing_fraction": float((starts < 0).mean()),
        "conftuner_confidence_token_fraction": float((in_confidence & valid).sum() / max(int(valid.sum()), 1)),
        "conftuner_reasoning_advantage_rms": float(np.sqrt(np.mean(np.square(reasoning)))),
        "conftuner_scale_factor": float(factor),
    }
    return advantages.astype(np.float32), diagnostics

