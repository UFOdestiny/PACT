"""PACT (Peers As Calibration Tests): ConfTuner's objective plus peer tests.

Every response states ``Confidence: NN%``.  The reasoning tokens receive the shared group-centred correctness credit
and the confidence block none (``method.conftuner``); the stated number is trained by the exact expected Brier score
over the percentage grid (``training/confidence_grid.py``), whose target is the response's own correctness.  The peer
tests add a centred residual penalty over peer strata: the responses of an update are grouped by the pass rate of
their peers, and gamma * Var_pi(r_s) penalises the spread of the strata's mean residuals
r_s = mean(confidence - correctness).  The penalty enters the grid loss as one linear coefficient per response
(``strata_linear``).

``method_options`` read here (the trainer reads the ``exact_*`` options and ``kl_log_ratio_clip``):

* ``lambda_agree``: the target is ``(1 - lambda) Y + lambda * K_{-i} / (G - 1)`` (0 everywhere but Peer-Target);
* ``strata_calibration`` (gamma), ramped in linearly over ``strata_warmup_steps`` updates;
* ``strata_bins``: number of equal-width strata (B);
* ``strata_center``: subtract the overall residual, so only the spread across strata is penalised;
* the stratum statistic: ``strata_peer_pass`` (leave-one-out peer pass rate, PACT), with ``strata_peer_pass_self``
  (the group pass rate including the response's own outcome, PACT-Self); ``strata_loo`` (leave-one-out answer
  agreement, PACT-Agree); ``strata_external`` (PACT-ExtDiff, a control that reads the external difficulty
  annotation of the training pool, ``labeler_correct / labeler_samples``);
* ``strata_shuffle``: permute the questions' peer strata across the update (PACT-Shuffled);
* ``strata_pool`` / ``strata_leave_question_out``: estimate the stratum residuals over the whole update, leaving the
  response's own question out (the host-method instances).

With ``strata_calibration: 0`` and ``lambda_agree: 0`` the objective is ConfTuner's.  ``peer_test_rewards`` and
``peer_test_coefficients`` add the same tests to methods whose confidence is trained otherwise (RLCR, DCPO, CREDO).
"""

from __future__ import annotations

import numpy as np

from method.conftuner import BLOCK_TAG, REWARD_FUNCS, REWARD_WEIGHTS, conftuner_advantages

__all__ = ["BLOCK_TAG", "REWARD_FUNCS", "REWARD_WEIGHTS", "advantages", "agreement_targets", "confidence_targets",
           "loo_agreement_strata", "peer_pass_strata", "peer_test_coefficients", "peer_test_rewards", "penalty_scale",
           "strata_linear", "strata_linear_pooled"]


def confidence_targets(labels: np.ndarray, *, lambda_agree: float, group_width: int) -> np.ndarray:
    """``(1 - lambda) Y + lambda p_loo(x)`` with the leave-one-out pass rate p_loo = K_{-i} / (G - 1)."""
    y = np.asarray(labels, dtype=np.float64)
    if not 0.0 <= lambda_agree <= 1.0:
        raise ValueError("lambda_agree must lie in [0, 1]")
    if lambda_agree == 0.0:
        return y
    if group_width < 2 or len(y) % group_width:
        raise ValueError("the leave-one-out target needs complete groups of at least two responses")
    grouped = y.reshape(-1, group_width)
    loo = ((grouped.sum(axis=1, keepdims=True) - grouped) / (group_width - 1)).reshape(-1)
    return (1.0 - lambda_agree) * y + lambda_agree * loo


def agreement_targets(keys, group_width: int) -> np.ndarray:
    """N_i / (G - 1): the share of a response's group peers with the same answer identity (label-free)."""
    keys = list(keys)
    if group_width < 2 or len(keys) % group_width:
        raise ValueError("agreement targets need complete groups of at least two responses")
    out = np.zeros(len(keys), dtype=np.float64)
    for start in range(0, len(keys), group_width):
        block = keys[start:start + group_width]
        for i, key in enumerate(block):
            out[start + i] = sum(1 for j, other in enumerate(block) if j != i and other == key) / (group_width - 1)
    return out


def loo_agreement_strata(keys, group_width: int, n_bins: int = 5) -> np.ndarray:
    """Per-response stratum from the mean pairwise agreement of the OTHER G - 1 answers of its group,
    t_bar_{-i} = (sum_j N_j - 2 N_i) / ((G - 1)(G - 2)), in n_bins equal bins: a response's own answer and label
    never enter its stratum."""
    G = int(group_width)
    if G < 3:
        raise ValueError("leave-one-out strata need groups of at least three responses")
    peers = agreement_targets(keys, G) * (G - 1)                 # N_i: group peers with the same answer identity
    pairs = peers.reshape(-1, G).sum(1).repeat(G)                # sum_b n_b (n_b - 1) of the group
    t = (pairs - 2.0 * peers) / ((G - 1) * (G - 2))
    return np.minimum(np.round(t * n_bins, 9).astype(np.int64), n_bins - 1)


def peer_pass_strata(labels, group_width: int, n_bins: int = 5, include_self: bool = False) -> np.ndarray:
    """Per-response stratum from the pass rate of the OTHER G - 1 responses of its group, K_{-i} / (G - 1), in n_bins
    equal bins.  The response's own outcome never enters its stratum, so E[h(K_{-i}) | X] is a degree-(G - 1)
    Bernstein polynomial of the question's pass rate p(X).  ``include_self`` is the same-sample control: the group
    pass rate K / G, which carries the response's own outcome."""
    y = np.asarray(labels, dtype=np.float64)
    G = int(group_width)
    if G < 2 or len(y) % G:
        raise ValueError("peer bins need complete groups of at least two responses")
    grouped = y.reshape(-1, G)
    if include_self:
        peers = np.repeat(grouped.mean(axis=1), G)
    else:
        peers = ((grouped.sum(axis=1, keepdims=True) - grouped) / (G - 1)).reshape(-1)
    return np.minimum(np.round(peers * n_bins, 9).astype(np.int64), n_bins - 1)


def strata_linear(verbal, labels, strata, gamma: float, center: bool = False) -> tuple[np.ndarray, dict]:
    """Per-response coefficient of the stratum penalty gamma sum_s (n_s / N)(m_bar_s - y_bar_s)^2.

    Its gradient in a response's stated confidence m_i is 2 gamma (m_bar_s - y_bar_s) / N, the same 1/N as the
    per-row Brier; m_bar_s is the mean sampled report of stratum s.  ``center`` subtracts the overall residual, so
    the penalty is gamma Var_pi(r_s) and the coefficients sum to zero: the overall level is left to the proper
    score.  Rows without a valid confidence get zero and do not enter the means.
    """
    q = np.asarray(verbal, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    strata = np.asarray(strata)
    valid = np.isfinite(q) & (q >= 0.0) & (q <= 1.0) & (strata >= 0)
    out = np.zeros(len(q), dtype=np.float64)
    residuals = {}
    shift = float(q[valid].mean() - y[valid].mean()) if (center and valid.any()) else 0.0
    for k in np.unique(strata[valid]):
        sel = valid & (strata == k)
        residuals[int(k)] = float(q[sel].mean() - y[sel].mean())
        out[sel] = 2.0 * gamma * (residuals[int(k)] - shift)
    info = {"pact_strata_bias_abs": float(np.mean(np.abs(list(residuals.values())))) if residuals else 0.0,
            "pact_strata_bias_low": residuals[min(residuals)] if residuals else 0.0,
            "pact_strata_bias_high": residuals[max(residuals)] if residuals else 0.0,
            "pact_strata_linear_abs": float(np.abs(out).mean())}
    return out, info


def strata_linear_pooled(verbal, labels, strata, gamma: float, *, group_width: int, bins: int, center: bool = True,
                         leave_question_out: bool = True, reduce_sum=None) -> tuple[np.ndarray, dict]:
    """``strata_linear`` with the stratum residuals pooled over the whole update and, optionally, leave-question-out.

    ``reduce_sum`` sums the per-stratum residual sums and counts over the processes, so every coefficient is
    estimated from the update's 64 questions whatever the number of GPUs.  With ``leave_question_out`` the
    coefficient of a response of question j uses only the other questions' responses: neither its own label nor
    its peers' residuals enter its test, so for fixed strata the applied coefficient does not depend on the
    question it is applied to.  Rows without a valid confidence get zero and do not enter the sums; a stratum
    that no other question occupies gives zero.
    """
    q = np.asarray(verbal, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    G, B = int(group_width), int(bins)
    if len(q) % G:
        raise ValueError("pooled strata need complete groups")
    if strata.size and (strata.min() < 0 or strata.max() >= B):
        raise ValueError("strata must lie in [0, bins)")
    valid = np.isfinite(q) & (q >= 0.0) & (q <= 1.0)
    question = np.repeat(np.arange(len(q) // G), G)
    sums = np.zeros((len(q) // G, B))
    counts = np.zeros((len(q) // G, B))
    np.add.at(sums, (question[valid], strata[valid]), (q - y)[valid])
    np.add.at(counts, (question[valid], strata[valid]), 1.0)
    total = np.concatenate([sums.sum(0), counts.sum(0)])
    if reduce_sum is not None:
        total = np.asarray(reduce_sum(total), dtype=np.float64)
    S, N = total[:B], total[B:]
    if leave_question_out:
        S_q, N_q = S[None, :] - sums, N[None, :] - counts          # other questions of the update only
    else:
        S_q, N_q = np.broadcast_to(S, sums.shape), np.broadcast_to(N, counts.shape)
    occupied = N_q > 0.5
    r_q = np.where(occupied, S_q / np.where(occupied, N_q, 1.0), 0.0)
    overall = S_q.sum(1) / np.maximum(N_q.sum(1), 1.0) if center else np.zeros(len(sums))
    coef = np.where(occupied, 2.0 * gamma * (r_q - overall[:, None]), 0.0)
    out = np.where(valid, coef[question, strata], 0.0)
    pooled = S[N > 0.5] / N[N > 0.5]
    info = {"pact_strata_bias_abs": float(np.abs(pooled).mean()) if pooled.size else 0.0,
            "pact_strata_bias_low": float(pooled[0]) if pooled.size else 0.0,
            "pact_strata_bias_high": float(pooled[-1]) if pooled.size else 0.0,
            "pact_strata_linear_abs": float(np.abs(out).mean()),
            "pact_strata_pooled_rows": float(N.sum())}
    return out, info


def coefficients(options: dict, verbal, labels, strata, gamma: float, group_width: int, reduce_sum=None):
    """The applied per-response coefficients: per-process stratum means (default) or pooled over the update."""
    center = bool(options.get("strata_center", True))
    if bool(options.get("strata_pool", False)):
        return strata_linear_pooled(verbal, labels, strata, gamma, group_width=group_width,
                                    bins=int(options.get("strata_bins", 5)), center=center,
                                    leave_question_out=bool(options.get("strata_leave_question_out", False)),
                                    reduce_sum=reduce_sum)
    if bool(options.get("strata_leave_question_out", False)):
        raise ValueError("strata_leave_question_out requires strata_pool")
    return strata_linear(verbal, labels, strata, gamma, center=center)


def stratify(options: dict, labels: np.ndarray, keys, group_width: int, step: int, difficulty=None) -> np.ndarray:
    """The stratum of every response under ``options`` (see the module docstring); ``difficulty`` is the external
    labeller's pass rate per response, read only with ``strata_external``."""
    G = int(group_width)
    bins = int(options.get("strata_bins", 5))
    peer_pass = bool(options.get("strata_peer_pass", False))
    peer_self = bool(options.get("strata_peer_pass_self", False))
    loo = bool(options.get("strata_loo", False))
    shuffle = bool(options.get("strata_shuffle", False))
    external = bool(options.get("strata_external", False))
    if external:
        if peer_pass or loo or shuffle:
            raise ValueError("strata_external replaces the peer statistic and takes no other strata option")
        d = np.asarray(difficulty if difficulty is not None else [], dtype=np.float64)
        if d.shape != (len(labels),) or not np.isfinite(d).all() or (d < 0).any() or (d > 1).any():
            raise ValueError("strata_external needs the labeller pass rate in [0, 1] of every response")
        # the binning of peer_pass_strata, applied to the external pass rate
        return np.minimum(np.round(d * bins, 9).astype(np.int64), bins - 1)
    if peer_self and not peer_pass:
        raise ValueError("strata_peer_pass_self requires strata_peer_pass")
    if peer_pass == loo:
        raise ValueError("exactly one of strata_peer_pass, strata_loo and strata_external sets the strata")
    if shuffle and (loo or peer_self):
        raise ValueError("strata_shuffle applies to the leave-one-out peer strata only")
    if loo:
        return loo_agreement_strata(keys, G, bins)
    strata = peer_pass_strata(labels, G, bins, include_self=peer_self)
    if shuffle:
        # each question's vector of peer strata moves to another question of the update (PACT-Shuffled)
        rows = strata.reshape(-1, G)
        strata = rows[np.random.default_rng(int(step)).permutation(len(rows))].reshape(-1)
    return strata


def penalty_scale(options: dict, step: int) -> tuple[float, float | None]:
    """gamma_t = gamma * min(t / warm-up, 1) and the ramp (None without a warm-up or penalty)."""
    gamma = float(options.get("strata_calibration", 0.0))
    warm = int(options.get("strata_warmup_steps", 0))
    if gamma > 0.0 and warm > 0:
        # a strong penalty on a near-constant early forecast drives it to a one-hot grid distribution
        ramp = min(1.0, max(0.0, float(step) / warm))
        return gamma * ramp, ramp
    return gamma, None


def peer_test_rewards(options: dict, verbal, labels, group_width: int, step: int, *, weight: float,
                      keys=None, reduce_sum=None) -> tuple[np.ndarray, dict]:
    """The peer tests as a reward term, for methods that train a sampled confidence through a reward (RLCR, DCPO).

    The centred stratum penalty gamma V(f) enters the method's own calibration reward with that reward's Brier
    weight ``weight``: the per-response term is ``-weight * 2 gamma_t e_hat_{S_i} * C_i`` (``strata_linear``, the
    coefficients held fixed), so its expected policy gradient is that of ``-weight * gamma_t * V`` at fixed
    assignments, as the grid surrogate is for PACT.  The strata, centring, warm-up and per-process mini-batch
    are PACT's (``stratify``); the residual is always ``C_i - Y_i``, whatever target the method uses.  ``verbal``
    is NaN where the method's own calibration reward sees no valid confidence; those responses get zero.
    """
    q = np.asarray(verbal, dtype=np.float64)
    linear, info = peer_test_coefficients(options, q, labels, group_width, step, keys=keys, reduce_sum=reduce_sum)
    stated = np.where(np.isfinite(q), q, 0.0)
    term = -float(weight) * linear * stated
    if info:
        info["pact_test_reward_abs"] = float(np.abs(term).mean())
    return term, info


def peer_test_coefficients(options: dict, verbal, labels, group_width: int, step: int, *, keys=None,
                           reduce_sum=None) -> tuple[np.ndarray, dict]:
    """The per-response coefficient ``2 gamma_t e_hat_{S_i}`` of the peer tests (zero before the penalty starts or
    without a valid confidence): ``peer_test_rewards`` puts it in a reward; DCPO+PACT and CREDO+PACT apply it to the
    expected grid confidence and to the readout regression."""
    q = np.asarray(verbal, dtype=np.float64)
    gamma, ramp = penalty_scale(options, int(step))
    if gamma <= 0.0:
        return np.zeros(len(q), dtype=np.float64), {}
    y = np.asarray(labels, dtype=bool)
    strata = stratify(options, y, keys, int(group_width), int(step))
    linear, info = coefficients(options, q, y, strata, gamma, int(group_width), reduce_sum)
    info = dict(info)
    if ramp is not None:
        info["pact_strata_ramp"] = ramp
    return linear, info


def advantages(context):
    for name in ("block_start", "completion_lengths", "completion_width"):
        if getattr(context, name) is None:
            raise ValueError(f"pact needs context.{name}")
    options = context.options
    weights = dict(zip(REWARD_FUNCS, REWARD_WEIGHTS))
    labels = np.asarray(context.labels, dtype=bool)
    rewards = (weights["format"] * np.asarray(context.formats, dtype=np.float64)
               + weights["accuracy"] * labels.astype(np.float64))
    adv, diagnostics = conftuner_advantages(
        rewards, labels, context.block_start, context.completion_lengths,
        completion_width=int(context.completion_width), group_width=context.group_width,
        scale_match=bool(options.get("scale_match", True)), formats=context.formats,
        trained=context.trained, reduce_sum=context.reduce_sum)
    target = confidence_targets(labels, lambda_agree=float(options.get("lambda_agree", 0.0)),
                                group_width=int(context.group_width))
    context.extra["confidence_target"] = target
    gamma, ramp = penalty_scale(options, int(context.step))
    if ramp is not None:
        diagnostics["pact_strata_ramp"] = ramp
    if gamma > 0.0:
        strata = stratify(options, labels, context.keys, int(context.group_width), int(context.step),
                          difficulty=context.difficulty)
        linear, info = coefficients(options, context.verbal, labels, strata, gamma, int(context.group_width),
                                    context.reduce_sum)
        context.extra["confidence_linear"] = linear
        diagnostics.update(info)
    diagnostics["pact_target_mean"] = float(target.mean())
    diagnostics["pact_target_label_gap_mean"] = float(np.abs(target - labels).mean())
    return adv, diagnostics
