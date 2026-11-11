"""Method registry: a run's method ID -> the module that defines its reward and credit.

Every method module declares

* ``REWARD_FUNCS`` / ``REWARD_WEIGHTS`` -- the reward functions the GRPO stack computes for every response
  (``training/rewards.py`` implements them with the evaluation judge);
* optionally ``advantages(context) -> (advantages, diagnostics)`` -- replaces the group-centred GRPO advantage
  with the method's own credit; one value per response or a ``(responses, completion_width)`` token matrix;
* optionally ``BLOCK_TAG`` -- the marker that opens the confidence block of the response; the trainer then fills
  ``context.block_start`` with the token index at which that block starts (-1 when absent);
* optionally ``NEEDS_REFERENCE_MODEL`` / ``NEEDS_POLICY_LOGPROBS``.

A method ID maps to its family through ``FAMILIES``: the exact name, else the longest registered prefix followed by
``-`` (``pact-agree`` -> ``pact``, ``rlcr-pact`` -> ``rlcr``).  ``method_options`` select the code path.  PACT and
ConfTuner's objective live in ``method/``; the other baselines are looked up in the optional ``baseline`` package,
which is not part of this release.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType

import numpy as np

BASELINES = ("rlvr", "rlcr", "capo", "dcpo", "credo")

# Registered families: method-ID prefix -> family (module name).
FAMILIES = {
    **{name: name for name in BASELINES},
    "base": "base",
    "pact": "pact",
    "conftuner": "pact",
    "conftuner-pact": "pact",
    "peer-target": "pact",
}
_PREFIXES = sorted(FAMILIES, key=len, reverse=True)


@dataclass
class AdvantageContext:
    """Everything a method's ``advantages`` hook may look at for one generation batch."""

    labels: np.ndarray                    # verified correctness per rollout (bool)
    formats: np.ndarray                   # format compliance per rollout (float 0/1)
    keys: list[str]                       # answer identity per rollout (judge-consistent)
    group_width: int                      # G
    step: int                             # optimizer step at generation time
    options: dict                         # method_options from the experiment YAML
    verbal: np.ndarray | None = None            # method-native confidence per rollout, NaN when absent/invalid
    completion_lengths: np.ndarray | None = None  # retained completion tokens per rollout
    completion_width: int | None = None         # padded completion length (columns of a token-level matrix)
    block_start: np.ndarray | None = None       # first token of the module's BLOCK_TAG block, -1 when absent
    mean_logprob: np.ndarray | None = None      # policy mean token log-probability per rollout
    reference_ppl: np.ndarray | None = None     # CAPO reference-model perplexity per rollout
    trained: np.ndarray | None = None           # rollout reaches the loss (not dropped as truncated)
    reduce_sum: object | None = None            # sums a numpy vector over processes (None on one process)
    difficulty: np.ndarray | None = None        # external labeller's pass rate of each rollout's question, NaN when
                                                # unlabelled (read only by the PACT-ExtDiff control)
    extra: dict = field(default_factory=dict)   # per-rollout outputs for the exact grid loss
                                                # (``confidence_target``, ``confidence_linear``)


def method_family(method: str) -> str:
    """The registered family of a run ID; an unregistered ID is returned unchanged."""
    if method in FAMILIES:
        return FAMILIES[method]
    for prefix in _PREFIXES:
        if method.startswith(prefix + "-"):
            return FAMILIES[prefix]
    return method


def load(method: str) -> ModuleType:
    name = method_family(method)
    if name in BASELINES:
        try:
            return importlib.import_module(f"baseline.implementation.{name}")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(f"{method}: the baseline implementations are not part of this release") from error
    return importlib.import_module(f"method.{name}")


def reward_specification(method: str) -> tuple[list[str], list[float]]:
    module = load(method)
    return list(module.REWARD_FUNCS), list(module.REWARD_WEIGHTS)


def needs_reference_model(method: str) -> bool:
    return bool(getattr(load(method), "NEEDS_REFERENCE_MODEL", False))


def needs_policy_logprobs(method: str) -> bool:
    return bool(getattr(load(method), "NEEDS_POLICY_LOGPROBS", False))


def block_tag(method: str) -> str | None:
    """The tag opening the confidence block of token-level methods, or None."""
    return getattr(load(method), "BLOCK_TAG", None)
