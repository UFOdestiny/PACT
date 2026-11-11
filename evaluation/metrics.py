"""Response-level accuracy, Brier, AUROC and 10-bin equal-width ECE."""
from __future__ import annotations

from typing import Any
import numpy as np

METRICS = ('acc', 'brier', 'auroc', 'ece')
ECE_BINS = 10


def ece_bins(p: np.ndarray, n_bins: int = ECE_BINS) -> np.ndarray:
    """Equal-width bins [k/B, (k+1)/B), the last one closed.  Rounding before the floor keeps
    two-decimal confidences on their own bin (np.linspace edges put 0.3 below 0.30000000000000004)."""
    return np.minimum(np.floor(np.round(np.asarray(p, dtype=float) * n_bins, 9)).astype(int), n_bins - 1)


def prepare(rows: list[dict[str, Any]], problems: list[str]) -> dict[str, np.ndarray]:
    indices = {problem: i for i, problem in enumerate(problems)}
    return {
        'labels': np.asarray([bool(r['answer_correct']) for r in rows], dtype=float),
        'probabilities': np.asarray([float(r['q_answer']) for r in rows]),
        'problem_indices': np.asarray([indices[str(r['problem_id'])] for r in rows]),
    }


def score(prepared: dict[str, np.ndarray], problem_weights: np.ndarray) -> dict:
    y, p = prepared['labels'], prepared['probabilities']
    w = problem_weights[prepared['problem_indices']].astype(float)
    total = w.sum()
    if total <= 0:
        raise ValueError('no observations with positive weight')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('confidence probabilities must lie in [0, 1]')
    positives = w @ y
    negatives = total - positives
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    pos = np.add.reduceat((w*y)[order], starts)
    neg = np.add.reduceat((w*(1-y))[order], starts)
    auc = (float(np.sum(pos * (np.cumsum(neg) - 0.5*neg)) / (positives*negatives))
           if positives > 0 and negatives > 0 else None)
    bins = ece_bins(p)
    ece = sum(abs(float(np.sum(w[bins == b] * (y[bins == b] - p[bins == b]))))
              for b in range(ECE_BINS)) / total
    return {'acc': float(positives/total), 'brier': float(w @ (p-y)**2 / total),
            'auroc': auc, 'ece': float(ece)}
