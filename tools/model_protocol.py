"""Backbone loading shared by training, evaluation and the data tools.

The protocol treats every backbone as a chat causal LM behind one tokenizer,
one system prompt and one vLLM engine.  The released checkpoints differ in
ways that would otherwise leak into several places:

* **Llama 3.1** templates carry a date header; the loader drops it so every prompt renders identically.
* **olmo3-8B** ships a hybrid thinking template whose switch defaults to thinking; the loader flips the default
  so every rendered prompt pre-fills the empty think block (non-thinking mode, ``force_non_thinking``).
* **Llama 3 / olmo3-8B** end a turn on a token that is not the
  tokenizer's ``eos_token`` (``<|eot_id|>``, ``<|im_end|>``); vLLM stops on the
  ``generation_config.json`` terminators, and the trainer builds completion masks from returned
  lengths and finish reasons.
"""

from __future__ import annotations

import json
import re
from os import PathLike
from pathlib import Path
from typing import Any

def read_model_config(model_name_or_path: str | PathLike[str]) -> dict:
    path = Path(model_name_or_path) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{model_name_or_path}: no config.json (only local backbones are supported)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_chat_tokenizer(
    model_name_or_path: str | PathLike[str], *, padding_side: str | None = None
) -> Any:
    from transformers import AutoTokenizer

    kwargs = {"padding_side": padding_side} if padding_side is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    if not tokenizer.chat_template:
        raise ValueError(f"{model_name_or_path}: tokenizer has no chat template")
    tokenizer.chat_template = force_non_thinking(drop_date_lines(tokenizer.chat_template))
    return tokenizer


def closes_think_block(tokenizer: Any) -> bool:
    """Detect a pre-closed reasoning container from the protocol's rendered generation prompt.

    Use the loaded tokenizer (including non-thinking normalization), not a model-name allowlist:
    saved checkpoints and local aliases must get the same decision as their backbone.
    """
    from tools.response_protocol import chat_messages

    prompt = tokenizer.apply_chat_template(
        chat_messages("rlcr", "math", "What is 1 + 1?"),
        add_generation_prompt=True, tokenize=False,
    )
    return re.search(r"</think>\s*\Z", prompt) is not None


DATE_LINES = ('"Cutting Knowledge Date:', '"Today Date:')


def drop_date_lines(template: str) -> str:
    """Remove the date header of the Llama 3.x templates.

    The Llama 3.1 / 3.2 templates always open with a system header carrying "Cutting Knowledge Date"
    and "Today Date" (3.2 reads the wall clock through ``strftime_now``, so the same prompt renders
    differently on different days).  Every path renders prompts through this tokenizer, and a
    checkpoint saves it, so training, rollouts and evaluation all see the same date-free prompt.
    Templates without those lines are returned unchanged.
    """
    return "\n".join(line for line in template.split("\n") if not line.lstrip("{- ").startswith(DATE_LINES))


THINKING_SWITCH = "{%- if enable_thinking is defined and enable_thinking is false %}"
NON_THINKING_DEFAULT = "{%- if enable_thinking is not defined or enable_thinking is false %}"


def force_non_thinking(template: str) -> str:
    """Make a hybrid thinking/non-thinking template (olmo3-8B) default to non-thinking.

    Those templates pre-fill an empty ``<think>\n\n</think>\n\n`` at the start of the assistant turn only when the
    caller passes ``enable_thinking=False``.  Every path here renders prompts with ``apply_chat_template`` and no
    extra arguments (training data, TRL's rollouts, evaluation, the saved checkpoint's template), so
    the switch is flipped in the template itself: the protocol runs the backbone in non-thinking mode,
    and passing ``enable_thinking=True`` explicitly still selects the thinking mode.  Templates without the switch
    are returned unchanged.
    """
    return template.replace(THINKING_SWITCH, NON_THINKING_DEFAULT)


def load_policy_model(model_name_or_path: str | PathLike[str], **kwargs: Any) -> Any:
    """Load a backbone as the trainable policy (or as a frozen reference model).

    ``kwargs`` are the usual ``from_pretrained`` arguments (``torch_dtype``,
    ``attn_implementation``, ``use_cache`` ...).  ``trust_remote_code`` and
    ``revision`` are accepted and ignored: every supported backbone loads with
    the native implementation from a local directory.
    """
    kwargs = dict(kwargs)
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("revision", None)
    if kwargs.get("config") is not None:
        raise ValueError("load_policy_model builds the config itself; pass model kwargs only")
    config = read_model_config(model_name_or_path)
    if config.get("quantization_config"):
        raise ValueError(
            f"{model_name_or_path} is a quantized checkpoint ({config['quantization_config'].get('quant_method')}); "
            "only bf16/fp32 local checkpoints are supported"
        )
    use_cache = kwargs.pop("use_cache", None)
    from transformers import AutoModelForCausalLM

    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
