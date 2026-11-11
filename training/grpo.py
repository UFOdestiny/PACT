"""The protocol's GRPO advantage, advantage RMS matching and the group-composition diagnostic.

Shared by every method module and by the trainer.
"""

from __future__ import annotations

import numpy as np


def grpo_advantages(formats: np.ndarray, labels: np.ndarray, *, group_width: int,
                    format_weight: float = 0.5, accuracy_weight: float = 0.5) -> np.ndarray:
    """The protocol's own advantage: the shared reward, centred on the group mean, no std scaling."""
    rewards = (format_weight * np.asarray(formats, dtype=np.float64)
               + accuracy_weight * np.asarray(labels, dtype=np.float64))
    grouped = rewards.reshape(-1, group_width)
    return (grouped - grouped.mean(axis=1, keepdims=True)).reshape(-1)


def match_rms(advantages: np.ndarray, reference: np.ndarray, *, mask: np.ndarray | None = None,
              reduce_sum=None) -> tuple[np.ndarray, float]:
    """Rescale ``advantages`` so their RMS over the trained entries equals the reference's.

    ``advantages`` is either per rollout or a token matrix; a per-rollout ``reference`` (the protocol's
    GRPO advantage) is broadcast to the same entries, so for a token matrix both RMS values are
    token-weighted exactly as the BNPO loss weights them.  ``mask`` selects the entries that reach the
    loss: valid tokens of rollouts that are not dropped as truncated.  ``reduce_sum`` sums the
    statistics over processes, so the factor is one scalar for the whole update rather than a
    per-GPU value.  This matches advantage RMS, not the optimizer's effective step; set
    ``scale_match: false`` to retain the method's own scale.
    """
    values = np.asarray(advantages, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if values.ndim == 2 and ref.ndim == 1:
        ref = np.broadcast_to(ref[:, None], values.shape)
    if ref.shape != values.shape:
        raise ValueError("reference must be per rollout or match the advantages")
    weights = np.ones_like(values) if mask is None else np.asarray(mask, dtype=np.float64)
    stats = np.asarray([(np.square(values) * weights).sum(), (np.square(ref) * weights).sum(), weights.sum()])
    if reduce_sum is not None:
        stats = np.asarray(reduce_sum(stats), dtype=np.float64)
    if stats[2] <= 0.0 or stats[0] <= 0.0 or stats[1] <= 0.0:
        # A zero own or reference RMS cannot define a nonzero scale match.
        return values, 1.0
    factor = float(np.sqrt(stats[1] / stats[0]))
    return values * factor, factor


def group_composition(labels: np.ndarray, *, group_width: int) -> dict[str, float]:
    """All-correct / all-wrong / mixed fractions and the mean pass rate of a batch.

    Pure group-centred binary correctness credit vanishes on homogeneous
    groups. Format/confidence rewards and KL can still contribute, so mixed
    is a label-composition diagnostic, not the fraction carrying any gradient.
    """
    grouped = np.asarray(labels, dtype=bool).reshape(-1, group_width)
    all_correct = grouped.all(axis=1)
    all_wrong = ~grouped.any(axis=1)
    return {
        "groups/all_correct": float(all_correct.mean()),
        "groups/all_wrong": float(all_wrong.mean()),
        "groups/mixed": float((~all_correct & ~all_wrong).mean()),
        "groups/pass_rate_mean": float(grouped.mean()),
        "groups/pass_rate_sd": float(grouped.mean(axis=1).std()),
    }
