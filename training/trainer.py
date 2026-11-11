"""Shared GRPO trainer: method dispatch, the exact grid loss, efficiency accounting.

Every method runs through this subclass of the vendored RLCR/TRL GRPO trainer.  The base class generates the group
of responses, computes the shared reward and the group-centred advantage.  This class then

* labels every response with the evaluation judge and logs the group composition (all-correct / all-wrong / mixed);
* hands the batch to the method's ``advantages`` hook when the method defines one (CAPO, DCPO, CREDO, ConfTuner,
  PACT) and swaps the advantages;
* with ``exact_confidence`` (ConfTuner, PACT, DCPO+PACT) adds the exact expected Brier score over the confidence grid
  (``training/confidence_grid.py``) and removes the sampled number tokens' own credit;
* with a ``readout`` confidence (CREDO) reads ``sigmoid(z_H - z_L)`` at each response's slot for the method's credit
  and adds the method's differentiable regression of that readout to the policy loss;
* with ``strata_calibration`` on a reward-trained confidence without a credit hook (RLCR+PACT) adds the peer-test
  term of ``method/pact.py::peer_test_rewards`` to the confidence block's tokens;
* applies the k3 KL guard ``kl_log_ratio_clip`` of the vendored loss when the method options set it;
* times generation and optimisation and logs tokens and memory (``training/efficiency.py``).
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import gather_object
from GRPO_Trainer import CustomTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from trl.trainer.utils import selective_log_softmax

from evaluation.generate import confidence_line, parse_confidence, readout_slot
from tools.model_protocol import load_chat_tokenizer
from tools.response_protocol import confidence_style, response_schema
from training import registry
from training.confidence_grid import GRIDS, grid_candidates, grid_loss_from_logits, hybrid_grid_loss
from training.efficiency import EfficiencySummaryCallback, StepTimer
from training.grpo import group_composition
from training.rewards import answer_identities, correctness_labels, format_reward

# The protocol's confidence-line tags (``evaluation.generate.confidence_line`` styles).
TAG_STYLES = {"CONFIDENCE:": "line", "Confidence:": "percent"}


def masked_mean_logp(per_token_logps, completion_mask):
    """Average aligned token log-probabilities over each retained completion."""
    if per_token_logps.shape != completion_mask.shape or per_token_logps.ndim != 2:
        raise ValueError("log probabilities and masks must be aligned matrices")
    if not bool(torch.isfinite(per_token_logps).all()):
        raise ValueError("log probabilities must be finite")
    weights = completion_mask.to(dtype=per_token_logps.dtype)
    lengths = weights.sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("every completion must retain at least one token")
    return (per_token_logps * weights).sum(dim=1) / lengths


def confidence_tag_span(text: str, tag: str) -> tuple[int, int] | None:
    """Character span of the block tag on the response's confidence line, or None.

    This is the match used by the parser and the format check (``evaluation.generate.confidence_line``: the last
    line-anchored, case-insensitive tag), so the token-level block starts on exactly the line whose value is
    scored; an earlier mid-line mention cannot open the block.
    """
    if tag not in TAG_STYLES:
        # A literal block tag (CREDO's ``<analysis>``): its last occurrence.
        start = text.rfind(tag)
        return None if start < 0 else (start, start + len(tag))
    match = confidence_line(text, TAG_STYLES[tag])
    return (match.start("tag"), match.end("tag")) if match else None


def block_token_start(tokenizer, token_ids: list[int], text: str, tag: str) -> int:
    """Index of the first completion token that belongs to the block opened by ``tag`` (-1 if absent).

    ``text`` is the decoded completion (special tokens skipped).  The token containing the tag's first
    character starts the block.  Decoding is monotone in the prefix length, so a binary search over
    prefixes finds the boundary in O(log n) decodes.
    """
    span = confidence_tag_span(text, tag)
    return -1 if span is None else token_index_at_char(tokenizer, token_ids, span[0])


def line_number_start(tokenizer, token_ids: list[int], text: str, tag: str) -> int:
    """Index of the token that starts the confidence number after ``tag`` and its separating spaces (-1 when no number
    follows, or when the number's first character shares a token with the space, as Llama-3's " 0": the ``line`` grid
    is read from the first digit; training/confidence_grid.py::line_candidates)."""
    span = confidence_tag_span(text, tag)
    if span is None:
        return -1
    match = re.match(r"[ \t]*(?=[.\d])", text[span[1]:])
    if match is None:
        return -1
    position = span[1] + match.end()
    index = token_index_at_char(tokenizer, token_ids, position)
    if index < 0 or not tokenizer.decode(token_ids[index:index + 1])[:1] == text[position]:
        return -1
    return index


def number_token_start(tokenizer, token_ids: list[int], text: str, tag: str) -> int:
    """Index of the first completion token after ``tag`` when a confidence number follows (-1 otherwise).

    The teacher-forced prefix ends exactly at the tag, and the grid candidates carry the separating
    space (" 83", training/confidence_grid.py::grid_candidates), so a report that skips the space or
    starts with another string (Llama-3's single token " .") is off-grid mass.
    """
    span = confidence_tag_span(text, tag)
    if span is None:
        return -1
    end = span[1]
    if end >= len(text) or re.match(r"[ \t]*[.\d]", text[end:]) is None:
        return -1
    return token_index_at_char(tokenizer, token_ids, end)


def tag_block_tokens(tokenizer, token_ids: list[int], text: str) -> tuple[int, int]:
    """Token range ``[start, end)`` of the last ``<confidence>...</confidence>`` block (the block RLCR's parser reads,
    ``evaluation.generate.parse_confidence``), ``(-1, -1)`` when absent or empty."""
    matches = list(re.finditer(r"<confidence>(.*?)</confidence>", text, re.DOTALL | re.IGNORECASE))
    if not matches or not matches[-1].group(1).strip():
        return -1, -1
    start, end = matches[-1].span()
    return token_index_at_char(tokenizer, token_ids, start), token_index_at_char(tokenizer, token_ids, end - 1) + 1


def token_index_at_char(tokenizer, token_ids: list[int], position: int) -> int:
    """Index of the token whose decoded span contains character ``position`` (binary search on prefixes)."""
    low, high = 1, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if len(tokenizer.decode(token_ids[:middle], skip_special_tokens=True)) > position:
            high = middle
        else:
            low = middle + 1
    return max(low - 1, 0)


def verbal_confidences(texts: list[str], method: str) -> np.ndarray:
    """Method-native confidence per response, NaN when unavailable or invalid."""
    style = confidence_style(method)
    values = ([parse_confidence(text, style=style, allow_percent=False) for text in texts]
              if style and style != "readout" else [None] * len(texts))
    return np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)


def raw_reference_per_token_logps(model, input_ids, attention_mask, logits_to_keep, *, batch_size):
    logps = []
    for start in range(0, input_ids.size(0), batch_size):
        ids = input_ids[start : start + batch_size]
        mask = attention_mask[start : start + batch_size]
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=logits_to_keep + 1).logits
        logps.append(selective_log_softmax(logits[:, :-1, :][:, -logits_to_keep:, :], ids[:, -logits_to_keep:]))
    return torch.cat(logps, dim=0)


class ExperimentTrainer(CustomTrainer):
    """The vendored GRPO trainer plus method dispatch (``PACT_METHOD`` / ``PACT_METHOD_OPTIONS``)."""

    # Weights are written in shards of this size, which keeps the host-memory peak of the terminal save near
    # 20 GB at 8B (one 16 GB file peaks near 48 GB on rank 0); HF and vLLM load sharded checkpoints alike.
    SAVE_MAX_SHARD_SIZE = "2GB"

    def __init__(self, *args, **kwargs):
        model = kwargs.get("model")
        if kwargs.get("processing_class") is None and isinstance(model, (str, os.PathLike)):
            kwargs["processing_class"] = load_chat_tokenizer(model, padding_side="left")
        super().__init__(*args, **kwargs)
        self.method = os.environ["PACT_METHOD"]
        self.method_options = json.loads(os.environ.get("PACT_METHOD_OPTIONS", "{}"))
        options = self.method_options
        module = registry.load(self.method)
        self.response_schema = response_schema(self.method)
        self.advantage_hook = getattr(module, "advantages", None)
        # Peer tests on a reward-trained confidence without a method credit hook (RLCR+PACT): the term of
        # method/pact.py::peer_test_rewards joins the confidence block's credit with the method's Brier weight.
        self.peer_test_weight = None
        if float(options.get("strata_calibration", 0.0)) > 0.0 and self.advantage_hook is None:
            weights = dict(zip(module.REWARD_FUNCS, module.REWARD_WEIGHTS))
            if "brier" not in weights:
                raise ValueError(f"{self.method}: peer tests need a Brier reward or an advantages hook")
            if options.get("strata_credit", "confidence") != "confidence":
                raise ValueError(f"{self.method}: reward-channel peer tests credit the confidence block")
            self.peer_test_weight = float(weights["brier"])
        self.block_tag = registry.block_tag(self.method)
        # Exact expected Brier score over the confidence grid (training/confidence_grid.py): its share
        # ``exact_weight``, the Brier weight ``calibration_weight``, the grid and its temperature.  The grid logits come
        # from the policy forward, so the score's gradient reaches the whole prefix; multi-token grids score their later
        # factors over a detached prefix cache.  The sampled number tokens lose their own credit.
        self.exact_confidence = bool(options.get("exact_confidence", False)) and self.block_tag is not None
        fixed = {"exact_full_gradient": True, "exact_normalize": "tokens", "exact_brier": True,
                 "exact_sampled_credit": "drop"}
        if self.exact_confidence and any(options.get(k, v) != v for k, v in fixed.items()):
            raise ValueError(f"{self.method}: the exact grid term takes {fixed}")
        self.exact_weight = float(options.get("exact_weight", 5.0))
        self.exact_calibration_weight = float(options.get("calibration_weight", 0.5))
        self.exact_grid = str(options.get("exact_grid", "percent"))
        if self.exact_grid not in GRIDS:
            raise ValueError(f"{self.method}: exact_grid must be one of {GRIDS}")
        temperature = options.get("exact_temperature")
        self.exact_temperature = float(self.temperature if temperature is None else temperature)
        # k3 KL guard (third_party/rlcr/GRPO_Trainer.py::k3_divergence): clamp the per-token log ratio of the KL
        # estimator to [-c, c].  Absent: the unclipped estimator.
        clip = options.get("kl_log_ratio_clip")
        self.kl_log_ratio_clip = None if clip is None else float(clip)
        # Readout confidence (CREDO): the two readout tokens; the method sets the regression target and weight.
        self.readout = confidence_style(self.method) == "readout"
        self.readout_tokens = None
        if self.readout:
            ids = self.processing_class.convert_tokens_to_ids(["<CONF HIGH>", "<CONF LOW>"])
            if any(t is None or t == self.processing_class.unk_token_id for t in ids):
                raise ValueError(f"{self.method}: the model has no readout tokens (baseline/credo_model.py)")
            self.readout_tokens = [int(t) for t in ids]
        self._readout_inputs = None
        self._exact_inputs = None
        self._exact_loss = None
        self._grid_candidates = None
        self.timer = StepTimer()
        metrics_path = Path(self.args.output_dir).resolve().parent / "metrics.jsonl"
        self.add_callback(EfficiencySummaryCallback(self.timer, metrics_path))
        self.reference_model = None
        if registry.needs_reference_model(self.method):
            # CAPO's reference perplexity: the starting policy.  TRL's KL reference (``self.ref_model``, beta > 0 under
            # ZeRO-2) is already that model on the device (a frozen copy of the initial bf16 / FlashAttention-2 policy),
            # so the perplexity reads it; a separate copy kept on the CPU adds ~15 GiB of host memory per process.
            if self.ref_model is None:
                raise ValueError(f"{self.method} needs the reference model: set beta > 0")
            self.reference_model = self.ref_model

    # --- exact expected score over the confidence grid --------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._exact_inputs = None
        self._exact_loss = None
        self._readout_inputs = None
        if self.readout and inputs.get("readout_slot") is not None:
            slots = inputs["readout_slot"]
            linear = inputs.get("readout_linear")
            self._readout_inputs = (slots, inputs["confidence_target"], inputs["readout_weight"],
                                    torch.zeros_like(inputs["readout_weight"]) if linear is None else linear)
            # The sampled slot token carries no policy-gradient credit: the readout's value is trained by the
            # regression alone (CREDO Sec. 3.3, "share parameters but no credit").
            adv = inputs["advantages"]
            if adv.dim() == 2:
                cols = torch.arange(adv.size(1), device=adv.device)[None, :]
                inputs = {**inputs, "advantages": adv * (cols != slots[:, None]).to(adv.dtype)}
        if self.exact_confidence and inputs.get("confidence_number_start") is not None:
            number_start = inputs["confidence_number_start"]
            # A method may replace the correctness label by its own per-response target (method/pact.py).
            target = inputs.get("confidence_target")
            self._exact_inputs = (number_start, inputs["correctness"] if target is None else target,
                                  inputs.get("confidence_linear"))
            # The sampled number tokens carry no REINFORCE credit: the exact term replaces it.
            adv = inputs["advantages"]
            if adv.dim() == 2:
                cols = torch.arange(adv.size(1), device=adv.device)[None, :]
                keep = (number_start[:, None] < 0) | (cols < number_start[:, None])
                inputs = {**inputs, "advantages": adv * keep.to(adv.dtype)}
        try:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
            if self._exact_loss is not None:
                loss = loss + self._exact_loss
            return loss
        finally:
            self._exact_inputs = None
            self._exact_loss = None
            self._readout_inputs = None

    def _log_exact(self, used: int, rows: int, loss: float, expected, entropies, masses, target) -> None:
        metrics = self._metrics["train"]
        metrics[f"{self.method}/exact_rows_fraction"].append(used / max(rows, 1))
        if expected:
            metrics[f"{self.method}/exact_expected_confidence"].append(float(np.mean(expected)))
            metrics[f"{self.method}/exact_loss_per_row"].append(loss / used)
            metrics[f"{self.method}/exact_grid_entropy"].append(float(np.mean(entropies)))
            metrics[f"{self.method}/exact_grid_mass"].append(float(np.mean(masses)))
            metrics[f"{self.method}/exact_target_mean"].append(float(target.float().mean()))

    def _full_gradient_exact(self, unwrapped, model, input_ids, attention_mask, logits_to_keep, *, batch_size, mode):
        """Policy forward with a hook on the head that keeps the number-position logits in the graph."""
        from transformers import DynamicCache
        if self._grid_candidates is None:
            self._grid_candidates = grid_candidates(self.processing_class, self.exact_grid)
        candidates, values = self._grid_candidates
        # Keep the first branching token in the policy graph. Digit-splitting tokenizers also
        # need conditional factors beyond this token, scored over a detached prefix cache.
        depth = 0
        while (depth < min(len(c) for c in candidates) - 1
               and all(c[depth] == candidates[0][depth] for c in candidates)):
            depth += 1
        common = tuple(candidates[0][:depth])
        single_token = all(len(c) == depth + 1 for c in candidates)
        grid_ids = torch.tensor([c[-1] for c in candidates], device=input_ids.device)
        number_start, target, linears = self._exact_inputs
        starts = number_start.to(input_ids.device)
        lengths = attention_mask[:, -logits_to_keep:].sum(dim=1)
        completion = input_ids[:, -logits_to_keep:]
        common_t = torch.tensor(common, dtype=completion.dtype, device=completion.device)
        captured = []
        offset = [0]

        def hook(module, args, output):
            rows = output.shape[0]
            for r in range(rows):
                i = offset[0] + r
                start = int(starts[i])
                pos = start + depth
                # the sampled response must follow the shared path (a report that skips the space is
                # off-grid mass and gets no exact credit, as on the trie path)
                if 0 < start and pos < int(lengths[i]) and bool((completion[i, start:pos] == common_t).all()):
                    # output[:, t] predicts completion token t (the parent drops the last logit)
                    captured.append((i, output[r, pos]))
            offset[0] += rows

        handle = unwrapped.lm_head.register_forward_hook(hook)
        try:
            logps = super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep,
                                                 batch_size=batch_size, mode=mode)
        finally:
            handle.remove()
        loss_sum = input_ids.new_zeros((), dtype=torch.float32)
        loss_value, expected, entropies, masses = 0.0, [], [], []
        full_cache = None
        was_training = unwrapped.training
        try:
            if not single_token and captured:
                # Eval mode keeps gradient checkpointing from dropping the cache; build it once per
                # micro-batch with all positions retained on sliding-window layers.
                unwrapped.eval()
                with torch.no_grad():
                    full_cache = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                                       past_key_values=DynamicCache(), logits_to_keep=1).past_key_values
            for i, logits in captured:
                kwargs = dict(temperature=self.exact_temperature,
                              calibration_weight=self.exact_calibration_weight,
                              linear=0.0 if linears is None else float(linears[i]))
                if single_token:
                    row_loss, mean, entropy, mass = grid_loss_from_logits(
                        logits, grid_ids, values, float(target[i]), **kwargs)
                else:
                    row_loss, mean, entropy, mass = hybrid_grid_loss(
                        logits, model, input_ids, attention_mask, candidates, values, float(target[i]),
                        full_cache=full_cache, row=i,
                        first=input_ids.shape[1] - logits_to_keep + int(starts[i]), depth=depth, **kwargs)
                loss_sum = loss_sum + row_loss
                loss_value += float(row_loss)
                expected.append(mean)
                entropies.append(entropy)
                masses.append(mass)
        finally:
            if not single_token and captured:
                unwrapped.train(was_training)
            del full_cache
        # BNPO scale: the sum over rows divided by the micro-batch's completion tokens.
        denominator = attention_mask[:, -logits_to_keep:].sum().clamp(min=1).to(loss_sum.dtype)
        self._exact_loss = self.exact_weight * loss_sum / denominator
        self._log_exact(len(expected), int(input_ids.shape[0]), loss_value, expected, entropies, masses, target)
        return logps

    def _readout_regression(self, unwrapped, model, input_ids, attention_mask, logits_to_keep, *, batch_size, mode):
        """Policy forward with a head hook that keeps the slot logits in the graph; the loss is
        ``weight * mean_V [(sigmoid(z_H - z_L) - target)^2 + linear sigmoid(z_H - z_L)]`` over the micro-batch rows with a
        retained slot (``linear``: the peer tests' coefficient for CREDO+PACT, else 0)."""
        slots, target, weight, linear = (t.to(input_ids.device) for t in self._readout_inputs)
        kept = attention_mask[:, -logits_to_keep:]
        high, low = self.readout_tokens
        captured = []
        offset = [0]

        def hook(module, args, output):
            for r in range(output.shape[0]):
                i = offset[0] + r
                slot = int(slots[i])
                # output[:, t] predicts completion token t (the parent drops the last logit)
                if 0 < slot < logits_to_keep and bool(kept[i, slot]):
                    captured.append((i, output[r, slot]))
            offset[0] += output.shape[0]

        handle = unwrapped.lm_head.register_forward_hook(hook)
        try:
            logps = super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep,
                                                 batch_size=batch_size, mode=mode)
        finally:
            handle.remove()
        if captured:
            rows = torch.tensor([i for i, _ in captured], device=input_ids.device)
            logits = torch.stack([z for _, z in captured]).float()
            readout = torch.sigmoid(logits[:, high] - logits[:, low])
            squared = (readout - target[rows].float()) ** 2
            self._exact_loss = (weight[rows].float() * (squared + linear[rows].float() * readout)).mean()
            metrics = self._metrics["train"]
            metrics[f"{self.method}/readout_loss_rows"].append(float(len(captured)) / max(int(input_ids.shape[0]), 1))
            metrics[f"{self.method}/readout_squared_error"].append(float(squared.mean()))
            metrics[f"{self.method}/readout_train_mean"].append(float(readout.mean()))
        return logps

    @torch.no_grad()
    def _readout_values(self, prompt_ids, prompt_mask, ids, mask, slots: np.ndarray) -> np.ndarray:
        """``sigmoid(z_H - z_L)`` at each response's slot from the current policy (NaN without a slot).

        The backbone runs on the rows with a slot and only the slot's hidden state meets the head, as
        ``h . (W_H - W_L)``; the inputs are the policy loss' own (left-padded prompt + completion)."""
        values = np.full(len(slots), np.nan, dtype=np.float64)
        rows = [i for i, slot in enumerate(slots) if slot > 0]
        if not rows:
            return values
        unwrapped = self.accelerator.unwrap_model(self.model)
        head = unwrapped.lm_head.weight
        high, low = self.readout_tokens
        direction = (head[high].float() - head[low].float())
        prompt_len = prompt_ids.size(1)
        step = int(self.args.per_device_train_batch_size)
        for start in range(0, len(rows), step):
            chunk = rows[start:start + step]
            width = max(int(slots[i]) for i in chunk)          # positions up to the last slot are enough
            input_ids = torch.cat([prompt_ids[chunk], ids[chunk, :width]], dim=1)
            attention = torch.cat([prompt_mask[chunk], mask[chunk, :width]], dim=1)
            hidden = unwrapped.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
            for r, i in enumerate(chunk):
                # the state at prompt_len + slot - 1 predicts completion token ``slot``
                values[i] = float(torch.sigmoid(hidden[r, prompt_len + int(slots[i]) - 1].float() @ direction))
            del hidden
        return values

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, mode="train"):
        if (self._readout_inputs is not None and model is not self.ref_model and torch.is_grad_enabled()
                and self.model.training and self._readout_inputs[0].shape[0] == input_ids.shape[0]):
            return self._readout_regression(self.accelerator.unwrap_model(model), model, input_ids, attention_mask,
                                            logits_to_keep, batch_size=batch_size, mode=mode)
        if (self._exact_inputs is None or model is self.ref_model or not torch.is_grad_enabled()
                or not self.model.training or self._exact_inputs[0].shape[0] != input_ids.shape[0]):
            return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep,
                                                batch_size=batch_size, mode=mode)
        return self._full_gradient_exact(self.accelerator.unwrap_model(model), model, input_ids, attention_mask,
                                         logits_to_keep, batch_size=batch_size, mode=mode)

    def _reduce_sum(self, values):
        """Sum a small numpy vector over all training processes (collective; every rank must call)."""
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float64), device=self.accelerator.device)
        return self.accelerator.reduce(tensor, reduction="sum").cpu().numpy()

    # --- efficiency accounting -------------------------------------------

    def training_step(self, *args, **kwargs):
        generation_before = self.timer.generation_s
        started = time.perf_counter()
        loss = super().training_step(*args, **kwargs)
        elapsed = time.perf_counter() - started
        # Generation happens inside training_step (colocated vLLM); keep the
        # optimizer bucket free of it.
        self.timer.add_optimizer(elapsed - (self.timer.generation_s - generation_before))
        return loss

    def log(self, logs, start_time=None):
        if self.model.training and "loss" in logs:
            logs = {**logs, **self.timer.step_metrics()}
        super().log(logs, start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        target = Path(output_dir or self.args.output_dir)
        terminal = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        if (not _internal_call and target.absolute() == Path(self.args.output_dir).absolute()
                and self.state.max_steps > 0 and self.state.global_step >= self.state.max_steps
                and ((terminal / "model.safetensors").is_file()
                     or (terminal / "model.safetensors.index.json").is_file())):
            return
        if self.is_deepspeed_enabled and self.deepspeed.config.get("zero_optimization", {}).get("stage") != 3:
            # ZeRO-1/2: every rank already holds the full bf16 weights on its GPU.  The library path first clones the
            # whole state dict to host memory on every rank at once (accelerate.get_state_dict), 4 x 16 GB at 8B on
            # top of vLLM's pinned backups.  Rank 0 saves straight from its GPU tensors, shard by shard; the other
            # ranks do nothing.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            if self.args.should_save:
                unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
                self._save(output_dir, state_dict=unwrapped.state_dict())
            return
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save(self, output_dir=None, state_dict=None):
        """``Trainer._save`` for the policy (a ``PreTrainedModel``), in ``SAVE_MAX_SHARD_SIZE`` shards."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
            output_dir, state_dict=state_dict, max_shard_size=self.SAVE_MAX_SHARD_SIZE)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _validate_group_layout(self, rows: list[dict]) -> None:
        """Every method computes its credit from ``reshape(-1, group_width)`` on the rows this
        process holds, so the process must hold whole groups in order.

        On more than one GPU the trainer splits the generation batch across processes and the
        advantage is sliced back contiguously after a gather, which only lines up if each
        process owns complete consecutive groups.  A wrong split would not raise anywhere -- the
        row count still divides by G -- it would silently mix two problems into one group, so it
        is checked here rather than trusted.  If this ever fires, the hook has to be moved behind
        the gather in ``GRPO_Trainer._generate_and_score_completions``.
        """
        width = int(self.num_generations)
        ids = [str(row.get("problem_id", "")) for row in rows]
        if len(ids) % width:
            raise ValueError(f"process holds {len(ids)} responses, not a whole number of groups of {width}")
        for start in range(0, len(ids), width):
            block = set(ids[start:start + width])
            if len(block) != 1:
                raise ValueError(
                    f"responses {start}-{start + width - 1} on this process span {len(block)} problems "
                    f"({sorted(block)}); the generation batch is not split into whole groups")

    def _record_prompt_trace(self, inputs: list[dict], *, eval: bool) -> None:
        if eval:
            return
        trace_path = os.environ.get("PACT_TRAIN_TRACE")
        if not trace_path:
            return
        # Each process holds its own slice of the update, so the trace is the union in rank order;
        # without the gather a four-GPU cell would record a quarter of the prompts it trained on.
        gathered = gather_object([[str(row.get("problem_id", "")) for row in inputs]])
        if not self.is_world_process_zero():
            return
        problem_ids, seen = [], set()
        for shard in gathered:
            for problem_id in shard:
                if problem_id and problem_id not in seen:
                    seen.add(problem_id)
                    problem_ids.append(problem_id)
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"optimizer_step": int(self.state.global_step) + 1,
                                     "problem_ids": problem_ids}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # --- generation, labels, method credit -----------------------------------

    def _generate_and_score_completions(self, inputs, eval=False):
        snapshot = copy.deepcopy(inputs)
        self._validate_group_layout(snapshot)
        self._record_prompt_trace(snapshot, eval=eval)
        started = time.perf_counter()
        result = super()._generate_and_score_completions(inputs, eval=eval)
        ids, mask = result["completion_ids"], result["completion_mask"]
        generated_mask = result.get("generated_mask", mask)
        tokens = [[int(t) for t, keep in zip(row.tolist(), valid.tolist()) if keep]
                  for row, valid in zip(ids, generated_mask, strict=True)]
        texts = self.processing_class.batch_decode(tokens, skip_special_tokens=True)
        golds = [row["answer"] for row in snapshot]
        sources = [row.get("source") for row in snapshot]
        labels = np.asarray(correctness_labels(texts, golds, sources), dtype=bool)
        mode = "train" if self.model.training else "eval"
        for key, value in group_composition(labels, group_width=self.num_generations).items():
            self._metrics[mode][key].append(float(value))

        if self.advantage_hook is not None:
            chats = [[{"role": "assistant", "content": text}] for text in texts]
            context = registry.AdvantageContext(
                labels=labels,
                formats=np.asarray(format_reward(self.response_schema, chats), dtype=np.float64),
                keys=answer_identities(texts, sources, group_width=int(self.num_generations)),
                group_width=int(self.num_generations),
                step=int(self.state.global_step),
                options=self.method_options,
                verbal=verbal_confidences(texts, self.method),
                completion_lengths=np.asarray([len(row) for row in tokens], dtype=np.int64),
                completion_width=int(ids.size(1)),
                # Rollouts that reach the loss: mask_truncated_completions zeroes the whole row of a
                # truncated completion, so scale statistics must not count it.
                trained=(mask.sum(dim=1) > 0).detach().cpu().numpy(),
                # One scale per update: every process reaches this hook during training.
                reduce_sum=(self._reduce_sum if self.model.training and not eval
                            and self.accelerator.num_processes > 1 else None),
                difficulty=np.asarray([row["labeler_correct"] / row["labeler_samples"] if row.get("labeler_samples")
                                       else np.nan for row in snapshot], dtype=np.float64),
            )
            if self.readout:
                # Truncated responses keep their slot here (the method's group statistics use them); the loss hook
                # skips slots outside the retained mask.
                slots = np.asarray([readout_slot(self.processing_class, row, text)
                                    for row, text in zip(tokens, texts, strict=True)], dtype=np.int64)
                context.verbal = self._readout_values(result["prompt_ids"], result["prompt_mask"], ids,
                                                      generated_mask, slots)
                result["readout_slot"] = torch.as_tensor(slots, dtype=torch.long, device=ids.device)
            if self.block_tag is not None:
                context.block_start = np.asarray(
                    [block_token_start(self.processing_class, row, text, self.block_tag)
                     for row, text in zip(tokens, texts, strict=True)], dtype=np.int64)
                if self.exact_confidence:
                    valid = np.isfinite(context.verbal) & (np.asarray(context.formats) > 0)
                    number_start = np.asarray(
                        [(line_number_start if self.exact_grid in ("line", "line99") else number_token_start)(
                            self.processing_class, row, text, self.block_tag) if ok else -1
                         for row, text, ok in zip(tokens, texts, valid, strict=True)], dtype=np.int64)
                    result["confidence_number_start"] = torch.as_tensor(number_start, dtype=torch.long,
                                                                        device=ids.device)
                    result["correctness"] = torch.as_tensor(labels.astype(np.float32), device=ids.device)
            # Statistics per response are taken over the generated tokens, not the loss mask:
            # a truncated completion is dropped from the loss but still has log probabilities.
            generated = generated_mask
            if registry.needs_policy_logprobs(self.method):
                context.mean_logprob = (masked_mean_logp(result["generation_per_token_logps"], generated)
                                        .detach().float().cpu().numpy())
            if self.reference_model is not None:
                with torch.no_grad():
                    reference = raw_reference_per_token_logps(
                        self.reference_model, torch.cat([result["prompt_ids"], ids], dim=1),
                        torch.cat([result["prompt_mask"], generated], dim=1), ids.size(1),
                        batch_size=self.args.per_device_train_batch_size,
                    )
                context.reference_ppl = (torch.exp(-masked_mean_logp(reference, generated))
                                         .detach().float().cpu().numpy())
                del reference
            advantages, diagnostics = self.advantage_hook(context)
            if "confidence_target" in context.extra:
                target = np.asarray(context.extra["confidence_target"], dtype=np.float32)
                if target.shape != (len(texts),) or not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
                    raise ValueError(f"{self.method}: confidence_target must be {len(texts)} values in [0, 1]")
                result["confidence_target"] = torch.as_tensor(target, device=ids.device)
            if "readout_weight" in context.extra:
                weight = np.asarray(context.extra["readout_weight"], dtype=np.float32)
                if weight.shape != (len(texts),) or not np.isfinite(weight).all() or (weight < 0).any():
                    raise ValueError(f"{self.method}: readout_weight must be {len(texts)} non-negative values")
                result["readout_weight"] = torch.as_tensor(weight, device=ids.device)
            if "readout_linear" in context.extra:
                linear = np.asarray(context.extra["readout_linear"], dtype=np.float32)
                if linear.shape != (len(texts),) or not np.isfinite(linear).all():
                    raise ValueError(f"{self.method}: readout_linear must be {len(texts)} finite values")
                result["readout_linear"] = torch.as_tensor(linear, device=ids.device)
            if "confidence_linear" in context.extra:
                # per-row linear term of the stratum penalty (method/pact.py::strata_linear)
                linear = np.asarray(context.extra["confidence_linear"], dtype=np.float32)
                if linear.shape != (len(texts),) or not np.isfinite(linear).all():
                    raise ValueError(f"{self.method}: confidence_linear must be {len(texts)} finite values")
                result["confidence_linear"] = torch.as_tensor(linear, device=ids.device)
            advantages = np.asarray(advantages, dtype=np.float32)
            if advantages.shape not in ((len(texts),), (len(texts), int(ids.size(1)))):
                raise ValueError(f"{self.method}: advantages must be ({len(texts)},) or "
                                 f"({len(texts)}, {int(ids.size(1))}), got {advantages.shape}")
            result["advantages"] = torch.as_tensor(advantages, dtype=torch.float32, device=ids.device)
            for key, value in diagnostics.items():
                self._metrics[mode][f"{self.method}/{key}"].append(float(value))
        elif self.peer_test_weight is not None:
            # The protocol advantage is the reward centred on its group (scale_rewards: false), so adding the
            # group-centred test term equals adding the term to the reward.
            if self.args.scale_rewards:
                raise ValueError("reward-channel peer tests assume scale_rewards: false")
            from method.pact import peer_test_rewards

            chats = [[{"role": "assistant", "content": text}] for text in texts]
            formats = np.asarray(format_reward(self.response_schema, chats), dtype=np.float64)
            verbal = np.where(formats > 0, verbal_confidences(texts, self.method), np.nan)
            G = int(self.num_generations)
            pooled_sum = (self._reduce_sum if self.model.training and not eval
                          and self.accelerator.num_processes > 1 else None)
            advantages = result["advantages"]
            if advantages.dim() != 1 or advantages.numel() != len(texts):
                raise ValueError(f"{self.method}: expected one advantage per response, got {tuple(advantages.shape)}")
            term, diagnostics = peer_test_rewards(self.method_options, verbal, labels, G, int(self.state.global_step),
                                                  weight=self.peer_test_weight, reduce_sum=pooled_sum)
            # The test term is credited to the confidence block only (the reasoning tokens keep the method's own
            # sequence advantage), centred over the group's rows that state a confidence.
            grouped = term.reshape(-1, G)
            spans = np.asarray([tag_block_tokens(self.processing_class, row, text) if np.isfinite(v) else (-1, -1)
                                for row, text, v in zip(tokens, texts, verbal, strict=True)], dtype=np.int64)
            has = (spans[:, 0] >= 0).reshape(-1, G)
            mean = (grouped * has).sum(axis=1, keepdims=True) / np.maximum(has.sum(axis=1, keepdims=True), 1)
            centred = np.where(has, grouped - mean, 0.0).reshape(-1)
            matrix = np.zeros((len(texts), int(ids.size(1))), dtype=np.float32)
            for row, (start, end) in enumerate(spans):
                if start >= 0:
                    matrix[row, start:end] = centred[row]
            result["advantages"] = advantages[:, None] + torch.as_tensor(matrix, dtype=advantages.dtype,
                                                                         device=advantages.device)
            kept = spans[:, 0] >= 0
            diagnostics["pact_test_block_tokens"] = float((spans[kept, 1] - spans[kept, 0]).mean()) if kept.any() else 0.0
            for key, value in diagnostics.items():
                self._metrics[mode][f"{self.method}/{key}"].append(float(value))

        if mode == "train":
            self.timer.add_generation(
                time.perf_counter() - started,
                completion_tokens=int(mask.sum().item()),
                prompt_tokens=int(result["prompt_mask"].sum().item()),
            )
            zero = result["advantages"].abs() < 1e-12
            if zero.dim() == 2:
                # Token-level credit: fraction of retained tokens that carry no advantage.
                valid = mask.bool()
                zero = zero[valid] if valid.any() else zero.flatten()
            self._metrics[mode]["groups/zero_advantage_fraction"].append(float(zero.float().mean().item()))
        return result
