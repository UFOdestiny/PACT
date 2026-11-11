"""The experiment's method-native response prompts and schemas.

The experiment controls questions, sampling, judging, and compute across
methods.  It does not force methods with different confidence channels through
one output language.  Each method is mapped to the response contract its
training signal actually consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseProtocol:
    schema: str
    math_prompt: str
    general_prompt: str
    choice_prompt: str
    short_answer_prompt: str
    confidence_style: str | None = None
    code_prompt: str = ""


BOXED_MATH = r"""Solve the math problem step by step. Put only the final answer in \boxed{} at the end."""

BOXED_GENERAL = r"""Solve the question step by step. Put only the final answer in \boxed{} at the end."""


# ConfTuner (official system prompt, src/llama_recipes/datasets2/*.py): "please provide your confidence
# (0%-100%) to your answer", written as "Confidence: 90%".
CONFTUNER_MATH = r"""Solve the math problem step by step. Put only the final answer in \boxed{}.
Finally, provide your confidence (0%-100%) that the answer is correct: on the final line write Confidence: followed by an integer percentage (e.g. Confidence: 80%) and no other text."""

CONFTUNER_GENERAL = r"""Solve the question step by step. Put only the final answer in \boxed{}.
Finally, provide your confidence (0%-100%) that the answer is correct: on the final line write Confidence: followed by an integer percentage (e.g. Confidence: 80%) and no other text."""


CODE_ANSWER = "Write the final answer as one complete, self-contained Python solution in a single ```python``` code block"


def code_prompt(general: str) -> str:
    """The general prompt for a coding task: a boxed contract puts the solution in one ```python``` block where it
    would box the answer (its confidence instruction unchanged); a tag contract keeps its tags and asks for the block
    inside ``<answer>``."""
    boxed = re.compile(r"Put only the final answer in \\boxed\{\}( at the end)?\.")
    if "put your final answer within \\boxed{}" in general:
        # CREDO: its coding prompt differs only in carrying the answer in a python block (Fan et al. 2026, B.1).
        return (general.replace("put your final answer within \\boxed{}", CODE_ANSWER[0].lower() + CODE_ANSWER[1:])
                .replace("{step-by-step solution with \\boxed{final answer}}",
                         "{step-by-step solution with the final ```python``` code block}"))
    if boxed.search(general):
        return boxed.sub(lambda m: CODE_ANSWER + (m.group(1) or "") + ".", general)
    return general + " For a coding task, the final answer inside <answer> </answer> must be one complete, " \
                     "self-contained Python solution in a single ```python``` code block."



def _protocol(schema: str, math: str, general: str, *, confidence_style: str | None = None) -> ResponseProtocol:
    return ResponseProtocol(
        schema=schema,
        math_prompt=math,
        general_prompt=general,
        choice_prompt=general + " For a multiple-choice question, the final answer must be only the option letter.",
        short_answer_prompt=general + " The final answer must be only the requested name or short phrase.",
        confidence_style=confidence_style,
        code_prompt=code_prompt(general),
    )


# PACT and ConfTuner keep ConfTuner's response contract; ``screening`` is the plain boxed contract of the difficulty
# annotation (tools/estimate_pass_rate.py).  The baselines' contracts are added from the optional ``baseline`` package.
PROTOCOLS = {
    "screening": _protocol("boxed", BOXED_MATH, BOXED_GENERAL),
    "pact": _protocol("conftuner", CONFTUNER_MATH, CONFTUNER_GENERAL, confidence_style="percent"),
}
try:
    from baseline.protocols import protocols as _baseline_protocols
except ModuleNotFoundError:
    pass
else:
    PROTOCOLS.update(_baseline_protocols(_protocol, BOXED_MATH, BOXED_GENERAL, CONFTUNER_MATH, CONFTUNER_GENERAL))

MATH_DATASETS = {"math", "pool_heldout", "gsm8k", "minerva_math", "aime2024", "amc2024"}
CHOICE_DATASETS = {"arc_challenge"}
SHORT_ANSWER_DATASETS = {"simpleqa"}
CODE_DATASETS = {"mbpp_plus_fenced"}


def method_name(method: str) -> str:
    """The method family whose response contract a run ID uses."""
    from training.registry import method_family

    return method_family(method)


def protocol(method: str) -> ResponseProtocol:
    name = method_name(method)
    try:
        return PROTOCOLS[name]
    except KeyError as error:
        raise ValueError(f"no response protocol for method: {method}") from error


def response_schema(method: str) -> str:
    return protocol(method).schema


def confidence_style(method: str) -> str | None:
    return protocol(method).confidence_style


def chat_messages(method: str, dataset: str, problem: str) -> list[dict[str, str]]:
    """The one conversation every path renders (training data, dev diagnostics, test generation).

    System message: the method's native instruction for the dataset type.  User message: the problem
    text only.  Keeping a system message avoids template-injected defaults (some templates insert a
    function-calling system prompt when none is given); the date lines of the Llama templates are
    removed when the tokenizer is loaded (tools/model_protocol.load_chat_tokenizer).
    """
    return [{"role": "system", "content": system_prompt(method, dataset)},
            {"role": "user", "content": str(problem).strip()}]


def system_prompt(method: str, dataset: str = "math") -> str:
    selected = protocol(method)
    if dataset in MATH_DATASETS:
        return selected.math_prompt
    if dataset in CHOICE_DATASETS:
        return selected.choice_prompt
    if dataset in SHORT_ANSWER_DATASETS:
        return selected.short_answer_prompt
    if dataset in CODE_DATASETS:
        return selected.code_prompt
    return selected.general_prompt
