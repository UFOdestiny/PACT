"""Exact expected Brier score over the percentage confidence grid.

The stated number is scored by enumeration instead of by its sampled value: every grid value ``v_k`` (0%..100%)
is written in the response's own format and teacher-forced on the response's prefix (through the confidence
tag), and the normalised grid distribution ``pi_k = p(v_k) / sum_j p(v_j)`` gives the loss

    L_i = w sum_k pi_k (v_k - Y_i)^2 = w [(m - Y_i)^2 + Var_pi(v)],    m = sum_k pi_k v_k.

The candidates are prefix-free token sequences (``grid_candidates`` checks it), so they form a token trie.  p(v_k) is
the probability that the reported number *starts with* that spelling; what follows it is not scored.

Two scorers compute the loss inside the policy forward, so its gradient reaches the whole prefix:

* ``grid_loss_from_logits`` (single-token grids: Llama 3) reads the logit vector at the number's position;
* ``hybrid_grid_loss`` (digit-splitting tokenizers: OLMo-3) keeps the first branching token in the policy graph
  and scores the remaining factors of the trie over a detached prefix cache.

Both add ``linear * m``: the per-row coefficient of PACT's stratum penalty (method/pact.py::strata_linear), whose
gradient in the grid mean is the penalty's.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

GRIDS = ("percent", "line99")


def splits_digits(tokenizer) -> bool:
    """True when the tokenizer writes a two-digit number with more tokens than a one-digit number (digit splitting:
    OLMo-3 writes " 85" as space, "8", "5"; Llama 3 writes it as space, "85")."""
    one = len(tokenizer(" 8", add_special_tokens=False)["input_ids"])
    two = len(tokenizer(" 85", add_special_tokens=False)["input_ids"])
    return two > one


def grid_candidates(tokenizer, grid: str = "percent") -> tuple[list[list[int]], list[float]]:
    """Token ids and values of the ``percent`` grid (" " + the integers 0-100, ConfTuner's "Confidence: 83%"), or of
    the ``line99`` grid (``line_candidates``).

    Each percent spelling is a single token after the space on Llama 3, so the grid distribution is exactly the
    official ConfTuner softmax over the 101 percentage tokens.  The prefix ends at the confidence tag
    (training/trainer.py::number_token_start), so the separating space belongs to the candidates; strings that skip
    it (Llama-3's " ." token in ": .5") are off-grid.  The candidates must be prefix-free so that their probabilities
    are disjoint events read off one token trie.  On a digit-splitting tokenizer (OLMo-3: " 85" is
    space, "8", "5") the bare percent spellings are not prefix-free ("8" starts "85", "10" starts "100"); every
    spelling that is a proper prefix of another one is then written with its "%" sign ("8%", "10%"), so p(8) is the
    probability of the exact report "8%" while p(85) stays the probability that the number starts with "85".  The trie
    shares the digit prefixes when scoring multi-token candidates.
    """
    if grid == "line99":
        return line_candidates(tokenizer)
    if grid != "percent":
        raise ValueError(f"unknown confidence grid: {grid}")
    spellings, values = [str(k) for k in range(101)], [k / 100 for k in range(101)]
    candidates = [list(tokenizer(" " + s, add_special_tokens=False)["input_ids"]) for s in spellings]
    seen = {tuple(c) for c in candidates}
    if splits_digits(tokenizer):
        prefixes = {k for k, c in enumerate(candidates)
                    if any(len(o) > len(c) and o[:len(c)] == c for o in candidates)}
        candidates = [list(tokenizer(" " + s + "%", add_special_tokens=False)["input_ids"]) if k in prefixes else c
                      for k, (s, c) in enumerate(zip(spellings, candidates))]
        seen = {tuple(c) for c in candidates}
    if len(seen) != len(candidates):
        raise ValueError(f"{grid} grid spellings collide after tokenisation")
    for c in candidates:
        if any(tuple(c[:n]) in seen for n in range(1, len(c))):
            raise ValueError(f"{grid} grid is not prefix-free after tokenisation: {c}")
    return candidates, values


def line_candidates(tokenizer, top: int = 99) -> tuple[list[list[int]], list[float]]:
    """Token ids and values of the ``line99`` grid: DCPO's "CONFIDENCE: 0.83" line, the two-decimal spellings
    0.00-0.99 read from the number's first digit (training/trainer.py::line_number_start; the separating space is in
    the prefix).  All spellings have four characters, so on a digit-splitting tokenizer (OLMo-3: "0", ".", "8", "3")
    they are prefix-free; p(v_k) is the probability that the report starts with that spelling ("0.9999" counts as
    0.99).  All of them share the path "0.", so the first branching token -- the one scored in the policy graph -- is
    the tenths digit, as the tens digit of the percent grid is for PACT.  A report of "1.00" is off-grid mass."""
    spellings = [f"{k / 100:.2f}" for k in range(top + 1)]
    candidates = [list(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in spellings]
    seen = {tuple(c) for c in candidates}
    if len(seen) != len(candidates):
        raise ValueError("line grid spellings collide after tokenisation")
    for c in candidates:
        if any(tuple(c[:n]) in seen for n in range(1, len(c))):
            raise ValueError(f"line grid is not prefix-free after tokenisation: {c}")
    return candidates, [k / 100 for k in range(top + 1)]


def candidate_trie(candidates: list[list[int]]):
    """The maximal internal paths of the candidate trie and, per candidate token, where it is scored.

    Returns ``(paths, rows, cols)``: ``paths`` are the internal nodes that are not a prefix of another
    internal node; token ``t`` of candidate ``k`` is predicted at position ``cols[k][t]`` (0 = the
    token before the number) of sequence ``rows[k][t]``, whose first ``t`` path tokens equal the
    candidate's.
    """
    internal = {tuple(c[:t]) for c in candidates for t in range(len(c))}
    paths = sorted(n for n in internal if not any(len(m) > len(n) and m[:len(n)] == n for m in internal))
    owner = {}
    for index, path in enumerate(paths):
        for t in range(len(path) + 1):
            owner.setdefault(path[:t], index)
    rows = [[owner[tuple(c[:t])] for t in range(len(c))] for c in candidates]
    cols = [list(range(len(c))) for c in candidates]
    return paths, rows, cols


def _prefix_cache(full_cache, row: int, length: int, repeats: int) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        k = layer.keys[row : row + 1, :, :length].detach().repeat_interleave(repeats, dim=0)
        v = layer.values[row : row + 1, :, :length].detach().repeat_interleave(repeats, dim=0)
        cache.update(k, v, layer_idx)
    return cache


def _trie_tensors(candidates, device):
    """Pack the shared trie layout once for all rows scored against this candidate set."""
    K = len(candidates)
    paths, rows, cols = candidate_trie(candidates)
    M, width = len(paths), max(len(p) for p in paths)
    path_ids = torch.zeros((M, width), dtype=torch.long, device=device)
    path_mask = torch.zeros((M, width), dtype=torch.long, device=device)
    for m, path in enumerate(paths):
        path_ids[m, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
        path_mask[m, : len(path)] = 1
    length = max(len(c) for c in candidates)
    tok = torch.zeros((K, length), dtype=torch.long, device=device)
    row = torch.zeros((K, length), dtype=torch.long, device=device)
    col = torch.zeros((K, length), dtype=torch.long, device=device)
    cand_mask = torch.zeros((K, length), dtype=torch.float32, device=device)
    for k, c in enumerate(candidates):
        tok[k, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)
        row[k, : len(c)] = torch.tensor(rows[k], dtype=torch.long, device=device)
        col[k, : len(c)] = torch.tensor(cols[k], dtype=torch.long, device=device)
        cand_mask[k, : len(c)] = 1.0
    return path_ids, path_mask, tok, row, col, cand_mask


def _cached_candidate_logps(model, input_ids, attention_mask, full_cache, i, first, trie, *,
                           temperature, skip_tokens=0):
    """Score trie factors after ``skip_tokens`` using a detached cache before the anchor token."""
    path_ids, path_mask, tok, row, col, cand_mask = trie
    M = path_ids.shape[0]
    # The anchor predicts the first candidate token; each path position predicts the next token.
    ids = torch.cat([input_ids[i : i + 1, first - 1 : first].expand(M, 1), path_ids], dim=1)
    mask = torch.cat([attention_mask[i : i + 1, : first - 1].expand(M, -1),
                      torch.ones((M, 1), dtype=torch.long, device=input_ids.device), path_mask], dim=1)
    past = _prefix_cache(full_cache, i, first - 1, M)
    logits = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True).logits / temperature
    logp = torch.log_softmax(logits.float(), dim=-1)[row[:, skip_tokens:], col[:, skip_tokens:],
                                                   tok[:, skip_tokens:]]
    return (logp * cand_mask[:, skip_tokens:]).sum(dim=1)


def hybrid_grid_loss(logits, model, input_ids, attention_mask, candidates, values, target, *, full_cache,
                     row, first, depth, temperature=1.0, calibration_weight=1.0, linear=0.0):
    """Score one multi-token grid with an in-graph first factor and cached remaining factors.

    ``first`` is the absolute position of the candidate's first token, before its shared path of
    length ``depth``. ``logits`` predicts the token at ``first + depth`` in the policy graph.
    The caller builds ``full_cache`` once per batch under no_grad with a plain DynamicCache and
    keeps the model in eval mode during cached forwards (checkpointing otherwise drops caches).
    These forwards retain gradients through the number positions, but their prefix KV is detached.
    Returns the same tuple as ``grid_loss_from_logits``; mass is conditional on the shared path.
    """
    suffixes = [c[depth:] for c in candidates]
    values = torch.as_tensor(values, device=logits.device, dtype=torch.float32)
    if values.numel() != len(candidates):
        raise ValueError("one value per grid candidate is required")
    first_ids = torch.tensor([c[0] for c in suffixes], device=logits.device)
    cand_logp = torch.log_softmax(logits.float() / temperature, dim=-1)[first_ids]
    if any(len(c) > 1 for c in suffixes):
        trie = _trie_tensors(suffixes, input_ids.device)
        cand_logp = cand_logp + _cached_candidate_logps(
            model, input_ids, attention_mask, full_cache, row, first + depth, trie,
            temperature=temperature, skip_tokens=1)
    logpi = torch.log_softmax(cand_logp, dim=0)
    pi = torch.exp(logpi)
    mean = (pi * values).sum()
    spread = (pi * (values - mean) ** 2).sum()
    row_loss = calibration_weight * ((mean - target) ** 2 + spread + float(linear) * mean)
    entropy = -(pi * logpi).sum()
    mass = torch.exp(torch.logsumexp(cand_logp, dim=0)).clamp(max=1.0)
    return row_loss, float(mean), float(entropy), float(mass)


def grid_loss_from_logits(logits, grid_ids, values, target, *, temperature=1.0, calibration_weight=1.0, linear=0.0):
    """The exact expected Brier score of one row from one full-vocabulary logit vector.

    ``logits`` is the policy forward's logit vector at the position that predicts the number's first token, taken
    *inside* the policy loss' own forward, so the term's gradient reaches every position of the prefix (question,
    reasoning, answer), as in the official ConfTuner code.  ``grid_ids`` are single tokens: the grid distribution is
    then exactly the softmax over the 101 grid tokens.  ``linear`` adds ``linear * mean``.  Returns
    ``(row_loss, expected_confidence, entropy, grid_mass)``.
    """
    logits = logits.float()
    grid = logits[grid_ids] / temperature
    logpi = torch.log_softmax(grid, dim=0)
    pi = torch.exp(logpi)
    values = torch.as_tensor(values, device=logits.device, dtype=torch.float32)
    mean = (pi * values).sum()
    # (m - target)^2 + Var_pi(v) is the exact expected Brier of the sampled report
    spread = (pi * (values - mean) ** 2).sum()
    row_loss = calibration_weight * ((mean - target) ** 2 + spread + float(linear) * mean)
    mass = torch.exp(torch.logsumexp(logits[grid_ids], dim=0) - torch.logsumexp(logits, dim=0)).clamp(max=1.0)
    entropy = -(pi * logpi).sum()
    return row_loss, float(mean), float(entropy), float(mass)
