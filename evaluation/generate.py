"""Generate response banks with each method's native response protocol."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from evaluation import code_exec, tasks
from tools.model_protocol import closes_think_block, load_chat_tokenizer
from tools.response_protocol import chat_messages, confidence_style, response_schema


def required_model_len(prompt_token_lengths: list[int], max_tokens: int) -> int:
    """vLLM context length: the requested output after the longest rendered prompt, at least 1,280 + max_tokens."""
    if not prompt_token_lengths:
        raise ValueError("at least one prompt is required")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return max(1024 + max_tokens + 256, max(prompt_token_lengths) + max_tokens + 32)


def parse_answer(text: str) -> str | None:
    """The final answer: the last ``<answer>`` block, else the last ``\\boxed{}``, else the last ``Answer:`` line."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip() or None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start >= 0:
        content_start = start + len(marker)
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:index].strip() or None
    answer_lines = re.findall(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", text)
    return answer_lines[-1].strip().strip("$") if answer_lines else None


CONFIDENCE_LINE = {"line": "CONFIDENCE", "percent": "Confidence"}


def confidence_line(text: str, style: str) -> re.Match | None:
    """The last line-anchored ``<word>:`` confidence line (case-insensitive), or None.

    Parsing, the format check and the trainer's token-level block boundary all use this one match,
    so a response is either scored, formatted and credited on the same line, or none of the three.
    Group ``tag`` spans the word and colon, group ``value`` the rest of the line.
    """
    word = CONFIDENCE_LINE[style]
    matches = list(re.finditer(rf"(?im)^[ \t]*(?P<tag>{word}[ \t]*:)[ \t]*(?P<value>.*?)[ \t]*$", text))
    return matches[-1] if matches else None


def parse_confidence(text: str, *, style: str = "tag", allow_percent: bool = True) -> float | None:
    """The stated confidence in [0, 1], or None.

    ``tag``: the last ``<confidence>`` block (RLCR); ``line``: the ``CONFIDENCE:`` line (DCPO); ``percent``: the
    ``Confidence: NN%`` line (ConfTuner, PACT).  ``allow_percent`` also reads a number in (1, 100] as a percentage
    and the first number of a non-numeric block (the trainer and the bank fields pass False).
    """
    if style == "tag":
        matches = re.findall(r"<confidence>(.*?)</confidence>", text, re.DOTALL | re.IGNORECASE)
        content = matches[-1].strip() if matches else None
    elif style in ("line", "percent"):
        match = confidence_line(text, style)
        content = match.group("value") if match else None
    else:
        raise ValueError(f"unknown confidence style: {style}")
    if not content:
        return None
    if style == "percent":
        # ConfTuner's contract: a percentage 0-100 followed by "%".  A decimal percentage ("44.1%") is read by its
        # value; the format check still requires the integer form.
        percent = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)[ \t]*%", content)
        return float(percent.group(1)) / 100.0 if percent and float(percent.group(1)) <= 100 else None
    try:
        value = float(content)
    except ValueError:
        if not allow_percent:
            return None
        first_number = re.search(r"-?\d+(?:\.\d+)?", content)
        if first_number is None:
            return None
        value = float(first_number.group())
    if 0.0 <= value <= 1.0:
        return value
    if allow_percent and 1.0 < value <= 100.0:
        return value / 100.0
    return None


# CREDO (baseline/implementation/credo.py): the response ends with exactly one analysis block and one confidence slot
# holding one readout token; the confidence is read from the two tokens' logits at the slot.
CREDO_SLOT = "<confidence>"
CREDO_TAIL = re.compile(r"<analysis>.+?</analysis>\s*<confidence>(?:<CONF HIGH>|<CONF LOW>)</confidence>\s*\Z",
                        re.DOTALL)


def readout_slot(tokenizer, token_ids: list[int], text: str) -> int:
    """Completion index of the first token inside the last ``<confidence>`` tag, or -1.

    ``text`` is the decoded completion (special tokens skipped).  The slot exists only when the tag ends exactly on
    a token boundary and a token follows it; the readout is the next-token distribution at that position.  Shared by
    the trainer and evaluation, so both read the same position.
    """
    tag = text.rfind(CREDO_SLOT)
    if tag < 0:
        return -1
    end = tag + len(CREDO_SLOT)
    low, high = 1, len(token_ids)
    while low < high:                      # first prefix whose decoded length exceeds ``end``
        middle = (low + high) // 2
        if len(tokenizer.decode(token_ids[:middle], skip_special_tokens=True)) > end:
            high = middle
        else:
            low = middle + 1
    index = low - 1
    if index < 1 or index >= len(token_ids):
        return -1
    if len(tokenizer.decode(token_ids[:index], skip_special_tokens=True)) != end:
        return -1
    return index


def credo_readouts(llm, tokenizer, requests) -> list[list[float | None]]:
    """CREDO's readout ``sigmoid(z_H - z_L)`` at the slot of every sampled response (None without a slot).

    Each response is teacher-forced up to its slot and one token is drawn twice, restricted to each readout token; in
    ``raw_logprobs`` mode vLLM returns the unrestricted log-softmax of the drawn token, so the two log-probabilities
    differ by exactly ``z_H - z_L`` (prefix caching makes the second request nearly free).
    """
    import math

    from vllm import SamplingParams, TokensPrompt

    tokens = tokenizer.convert_tokens_to_ids(["<CONF HIGH>", "<CONF LOW>"])
    if any(t is None or t == tokenizer.unk_token_id for t in tokens):
        raise ValueError("the model has no CREDO readout tokens (baseline/credo_model.py)")
    slots, prompts = [], []
    for request in requests:
        row = []
        for output in request.outputs:
            ids = list(output.token_ids)
            slot = readout_slot(tokenizer, ids, tokenizer.decode(ids, skip_special_tokens=True))
            row.append(slot)
            if slot >= 0:
                prompts.append(TokensPrompt(prompt_token_ids=list(request.prompt_token_ids) + ids[:slot]))
        slots.append(row)
    logps = []
    for token in tokens:
        params = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=0, allowed_token_ids=[token])
        outputs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
        logps.append([float(out.outputs[0].logprobs[0][token].logprob) for out in outputs])
    values = iter(1.0 / (1.0 + math.exp(low - high)) for high, low in zip(*logps, strict=True))
    return [[next(values) if slot >= 0 else None for slot in row] for row in slots]


def parse_dataset_answer(text: str, dataset: str, row: dict | None = None) -> str | None:
    """Canonical answer for a dataset, or None when the response has none (code: the extracted program)."""
    raw = code_exec.extract_code(text) if dataset in tasks.CODE_DATASETS else parse_answer(text)
    return None if raw is None else tasks.parse(dataset, raw, row)


def format_compliant(text: str, schema: str, *, think_closed: bool | None = None) -> bool:
    """Apply the native gate, optionally counting the prompt's closed think block.

    ``think_closed`` (the chat template pre-fills a closed ``<think>`` block, OLMo-3) lets the RLCR / RLVR
    patterns accept a response without its own ``</think>``.  ``None`` reads ``PACT_THINK_CLOSED``, which the
    launcher derives from the tokenizer, so the reward functions and the dev diagnostic need no extra argument;
    evaluation detects it from its own tokenizer and passes it explicitly, and ``score_response`` records it in
    the bank row.
    """
    if think_closed is None:
        think_closed = os.environ.get("PACT_THINK_CLOSED") == "1"
    # RLCR's official reward_fns.format_reward patterns ("tabc" for RLCR, "ta" for its RLVR comparator),
    # applied with re.match: nothing is required before </think>, and a repeated tag is not rejected
    # (the last <answer> / <confidence> is the one read).
    patterns = {
        "rlcr": r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z",
        "reasoning_answer": r".*?</think>\s*<answer>.*?</answer>\s*\Z",
    }
    if think_closed:
        patterns = {
            "rlcr": r"\s*(?:.*?</think>)?\s*.*?<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z",
            "reasoning_answer": r"\s*(?:.*?</think>)?\s*.*?<answer>.*?</answer>\s*\Z",
        }
    if schema == "credo":
        return (text.count(r"\boxed{") == 1 and parse_answer(text) is not None
                and text.count("<analysis>") == 1 and text.count(CREDO_SLOT) == 1
                and text.find(r"\boxed{") < text.find("<analysis>") and CREDO_TAIL.search(text) is not None)
    if schema == "boxed":
        return text.count(r"\boxed{") == 1 and parse_answer(text) is not None
    if schema in ("dcpo", "conftuner"):
        style = "line" if schema == "dcpo" else "percent"
        word = CONFIDENCE_LINE[style]
        return (text.count(r"\boxed{") == 1 and parse_answer(text) is not None
                and len(re.findall(rf"(?im)^[ \t]*{word}[ \t]*:", text)) == 1
                and parse_confidence(text, style=style, allow_percent=False) is not None
                and (style != "percent"
                     or re.fullmatch(r"\d{1,3}[ \t]*%", confidence_line(text, style).group("value")) is not None))
    try:
        return bool(re.match(patterns[schema], text, re.DOTALL | re.MULTILINE))
    except KeyError as error:
        raise ValueError(f"unknown response schema: {schema}") from error


def selected_logprobs(output) -> list[float | None]:
    values = []
    for token_id, candidates in zip(output.token_ids, output.logprobs or []):
        item = candidates.get(token_id)
        values.append(float(item.logprob) if item is not None else None)
    return values


def score_response(
    text: str, dataset: str, row: dict, *, method: str, think_closed: bool | None = None, judge=None,
    readout: float | None = None,
) -> dict:
    """Every judged field of one response; shared by evaluation and dev diagnostics.  ``judge`` maps the
    canonical answer to ``(verdict, stage)``; ``score_responses`` passes one that remembers repeats.  A ``readout``
    method (CREDO) takes its confidence from ``readout`` (``credo_readouts``) instead of the text."""
    schema = response_schema(method)
    answer = parse_dataset_answer(text, dataset, row)
    style = confidence_style(method)
    if style == "readout":
        confidence = readout
    else:
        confidence = parse_confidence(text, style=style, allow_percent=False) if style else None
    if think_closed is None:
        think_closed = os.environ.get("PACT_THINK_CLOSED") == "1"
    valid_format = format_compliant(text, schema, think_closed=think_closed)
    verdict, stage = (judge(answer) if judge is not None
                      else tasks.judge_with_stage(dataset, answer, str(row["answer"]), row))
    extra = {"think_closed": True} if think_closed and schema in ("rlcr", "reasoning_answer") else {}
    return {
        **extra,
        "parsed_answer": answer,
        "gold_answer": str(row["answer"]),
        "answer_correct": verdict,
        "judge_stage": stage,
        "answer_parse_valid": answer is not None,
        "q_answer": confidence if confidence is not None else 0.0,
        "q_answer_parse_valid": confidence is not None,
        "response_schema": schema,
        "format_compliant": valid_format,
        "full_schema_compliant": bool(
            answer is not None and valid_format and (style is None or confidence is not None)
        ),
    }


def score_responses(
    texts: list[str], dataset: str, row: dict, *, method: str, think_closed: bool | None = None,
    readouts: list[float | None] | None = None,
) -> list[dict]:
    """``score_response`` for the samples of one prompt; an answer repeated across them is judged once."""
    judged: list[tuple[str | None, tuple[bool, str]]] = []
    readings = tasks.Readings()

    def judge(answer):
        for seen, verdict in judged:
            if seen == answer:
                return verdict
        before = tasks.timeouts()
        verdict = tasks.judge_with_stage(dataset, answer, str(row["answer"]), row, readings)
        if tasks.timeouts() == before:
            judged.append((answer, verdict))
        return verdict

    readouts = [None] * len(texts) if readouts is None else readouts
    return [score_response(text, dataset, row, method=method, think_closed=think_closed, judge=judge,
                           readout=readout)
            for text, readout in zip(texts, readouts, strict=True)]


def _score_job(job):
    texts, dataset, row, method, think_closed, readouts = job
    return score_responses(texts, dataset, row, method=method, think_closed=think_closed, readouts=readouts)


def score_requests(jobs: list[tuple], workers: int) -> list[list[dict]]:
    """``_score_job`` per prompt, in order, over ``workers`` spawned CPU processes (never forked from vLLM)."""
    if workers <= 1:
        return [_score_job(job) for job in jobs]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_score_job, jobs, chunksize=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    records = pq.read_table(args.data).to_pylist()
    tokenizer = load_chat_tokenizer(args.model)
    think_closed = closes_think_block(tokenizer)
    prompts = [
        tokenizer.apply_chat_template(chat_messages(args.method, args.dataset, row["problem"]),
                                      add_generation_prompt=True, tokenize=False)
        for row in records
    ]
    model_len = required_model_len([len(tokenizer.encode(prompt)) for prompt in prompts], args.max_tokens)
    started = time.time()
    llm = LLM(
        model=str(args.model), tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
        logprobs_mode="raw_logprobs", max_model_len=model_len,
    )
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed, logprobs=1)
    load_seconds = time.time() - started
    generation_started = time.time()
    outputs = llm.generate(prompts, params)
    generation_seconds = time.time() - generation_started
    readouts = ([None] * len(outputs) if confidence_style(args.method) != "readout"
                else credo_readouts(llm, tokenizer, outputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".jsonl.tmp")
    total_completion_tokens = 0
    scored = score_requests(
        [([output.text for output in request.outputs], args.dataset, row, args.method, think_closed, readout)
         for row, request, readout in zip(records, outputs, readouts, strict=True)],
        int(os.environ.get("PACT_JUDGE_WORKERS", "1")))
    with temporary.open("w") as handle:
        for row, prompt, request, judged in zip(records, prompts, outputs, scored, strict=True):
            for sample_id, (output, fields) in enumerate(zip(request.outputs, judged, strict=True)):
                total_completion_tokens += len(output.token_ids)
                result = {
                    "dataset": args.dataset,
                    "problem_id": str(row["problem_id"]),
                    "method": args.method,
                    "seed": args.seed,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "response": output.text,
                    "problem": str(row["problem"]),
                    **fields,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "completion_tokens": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                    "generation_max_tokens": args.max_tokens,
                    "token_ids": list(output.token_ids),
                    "token_logprobs": selected_logprobs(output),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    if think_closed:
        config["think_closed"] = True
    (args.output.parent / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output.parent / "resources.json").write_text(
        json.dumps(
            {
                "wall_seconds": time.time() - started,
                "model_load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "gpus": args.tensor_parallel_size,
                "generation_requests": len(records),
                "samples_per_request": args.n,
                "total_completion_tokens": total_completion_tokens,
                "completion_tokens_per_second": total_completion_tokens / max(generation_seconds, 1e-9),
                "requested_model_len": model_len,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
