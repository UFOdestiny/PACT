"""LLM re-judge of short-answer and multiple-choice banks with RLCR's official judge.

The judge is RLCR's ``llm_confidence_verifier`` (``eval/check_functions.py`` of the official RLCR code; RLCR paper
Section B.4): Llama-3.1-8B-Instruct (here the local ``resource/models/Llama-3.1-8B-Instruct``) under its own chat
template, RLCR's system prompt and question / ground-truth / model-answer message verbatim, greedy decoding
(temperature 0, 20 tokens), and "correct" when the reply contains "yes" (case-insensitive).  The judge sees only the
extracted final answer, never the confidence, under two extraction rules:

* strict (``llm_correct``): ``evaluation.generate.parse_answer`` -- the last ``<answer>`` block, else ``\\boxed{}``,
  else an ``Answer:`` line, as the live judge reads it;
* lenient (``llm_correct_lenient``, the paper's SimpleQA label): the strict answer, else, for a response that
  stopped (not truncated), its last line outside the confidence report (``<analysis>`` / ``<confidence>`` blocks and
  ``Confidence:`` / ``CONFIDENCE:`` lines removed, after the last ``</think>``).  ConfTuner-contract models often
  answer a factual question in plain text (e.g. "Rudolf Hess.\\n\\nConfidence: 63%"), which the strict
  rule cannot read; this is the short-answer counterpart of the lenient multiple-choice rule (evaluation.multiple_choice).

A response without an answer is wrong without a query (RLCR sends "I don't know"); an answer longer than 4,000
characters is cut to that length.  Ground truth: a short-answer set gives its alias list (one alias, as SimpleQA has: the
string), a multiple-choice set gives "<letter>. <option text>" with the options in the question.

Identical (question, prediction) pairs across all the given banks of a dataset are judged once: the pairs are sorted
and equal neighbours (``==``) share one query.  Read-only on the banks: each judged bank gets ``llm_judge.jsonl``
(one row per response: problem_id, sample_id, prediction, prediction_lenient, judge_output, judge_output_lenient,
llm_correct, llm_correct_lenient, rule_correct) and ``llm_judge.json`` (judge settings, the bank's ``native.jsonl``
size and mtime, agreement of both LLM labels with the rule judge and, for multiple choice, with evaluation.multiple_choice's
lenient letter judge) next to ``native.jsonl``.  Only complete banks
(``completion.json`` matching ``native.jsonl``) are read; a bank whose ``llm_judge.json`` matches its native file is
skipped unless ``--force``.

    CUDA_VISIBLE_DEVICES=0 python -m evaluation.llm_judge --datasets simpleqa \\
        --runs artifacts/llama-3.1-8b-instruct/conftuner-pact/seed-41
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.multiple_choice import REPORT_LINE, lenient_letter
from evaluation.generate import parse_answer
from evaluation.tasks import CHOICE_DATASETS, SHORT_ANSWER_DATASETS
from training.experiment import RESOURCE, cell_path, data_path

JUDGE_MODEL = Path(os.environ.get("PACT_JUDGE_MODEL") or RESOURCE / "models/Llama-3.1-8B-Instruct")
# RLCR eval/check_functions.py, llm_confidence_verifier, verbatim (whitespace included).
SYSTEM_PROMPT = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers.
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not.
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """
USER_TEMPLATE = """
            Question: {question}
            Ground Truth Answers: {gold}
            Model Generated Answer: {prediction}
            """
MAX_TOKENS = 20
MODEL_LEN = 16384
PREDICTION_CHARS = 4000   # longer extracted answers are cut to this many characters before judging


def bank_path(run: Path, dataset: str) -> Path:
    config = yaml.safe_load((run / "experiment.yaml").read_text())
    return cell_path(config, int(run.name.split("-")[-1]), dataset)


def bank_complete(path: Path) -> bool:
    """``completion.json`` was written for the current ``native.jsonl`` (the bank is not being written)."""
    try:
        completion = json.loads((path / "completion.json").read_text())
        stat = (path / "native.jsonl").stat()
        return completion["native_bytes"] == stat.st_size and completion["native_mtime_ns"] == stat.st_mtime_ns
    except (OSError, ValueError, KeyError):
        return False


def ground_truth(dataset: str, row: dict) -> str:
    if dataset in CHOICE_DATASETS:
        letter = str(row["answer"]).strip().upper()
        return f"{letter}. {list(row['choices'])[ord(letter) - ord('A')]}"
    aliases = [str(a) for a in (row.get("aliases") if row.get("aliases") is not None else [])]
    return str(aliases) if len(aliases) > 1 else (aliases[0] if aliases else str(row["answer"]))


def messages(question: str, gold: str, prediction: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question, gold=gold, prediction=prediction)}]


def verdict(text: str) -> bool:
    return "yes" in text.lower()


def last_line(response: str, finish_reason: str | None) -> str | None:
    """The lenient fallback answer: the last line of a finished response outside its confidence report."""
    if finish_reason != "stop":
        return None
    text = re.split(r"(?i)</think>", response)[-1]
    text = re.sub(r"(?is)<(analysis|confidence)>.*?(?:</\1>|\Z)", "\n", text)
    lines = [ln.strip().strip("*$ ").strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not REPORT_LINE.match(ln)]
    return lines[-1] if lines else None


def read_bank(path: Path, dataset: str, rows_by_problem: dict) -> list[dict]:
    """One record per response: identifiers, the extracted answer and the rule-based labels."""
    out = []
    with (path / "native.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            source = rows_by_problem[str(r["problem_id"])]
            strict = parse_answer(r["response"])
            record = {"problem_id": str(r["problem_id"]), "sample_id": int(r["sample_id"]), "prediction": strict,
                      "prediction_lenient": strict if strict is not None else last_line(r["response"], r.get("finish_reason")),
                      "rule_correct": bool(r["answer_correct"])}
            if dataset in CHOICE_DATASETS:
                letter = lenient_letter(r, list(source["choices"]))
                record["letter_correct"] = letter == str(source["answer"]).strip().upper()
            out.append(record)
    return out


FIELDS = ("prediction", "prediction_lenient")


def judge_records(dataset: str, banks: list[list[dict]], rows_by_problem: dict, generate) -> list[list[dict]]:
    """Per record of every bank, ``{field: (judge_output, correct)}`` for both extraction rules (``FIELDS``).
    ``generate`` maps a list of chat conversations to the judge's reply texts.  Equal (problem, prediction) pairs,
    over both rules and all banks, share one query."""
    items = [(record["problem_id"], record[field], b, i, field) for b, bank in enumerate(banks)
             for i, record in enumerate(bank) for field in FIELDS if record[field] is not None]
    items.sort(key=lambda item: (item[0], item[1]))
    queries, owner = [], []
    for k, (problem_id, prediction, _, _, _) in enumerate(items):
        if k == 0 or not (items[k - 1][0] == problem_id and items[k - 1][1] == prediction):
            source = rows_by_problem[problem_id]
            queries.append(messages(str(source["problem"]), ground_truth(dataset, source),
                                    prediction[:PREDICTION_CHARS]))
        owner.append(len(queries) - 1)
    replies = generate(queries) if queries else []
    if len(replies) != len(queries):
        raise ValueError(f"{len(replies)} judge replies for {len(queries)} queries")
    out = [[{field: ("", False) for field in FIELDS} for _ in bank] for bank in banks]
    for (_, _, b, i, field), q in zip(items, owner, strict=True):
        out[b][i][field] = (replies[q], verdict(replies[q]))
    answered = [sum(r[field] is not None for bank in banks for r in bank) for field in FIELDS]
    print(f"{dataset}: {sum(map(len, banks))} responses, {answered[0]} strict / {answered[1]} lenient answers, "
          f"{len(queries)} distinct judge queries", flush=True)
    return out


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    observed = float(np.mean(a == b))
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return {"agreement": observed, "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "llm_only": int(np.sum(a & ~b)), "other_only": int(np.sum(~a & b)),
            "llm_acc": float(a.mean()), "other_acc": float(b.mean())}


def vllm_generate(model: Path):
    """The judge on the visible GPU: RLCR's settings, prompts rendered with the model's own chat template."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model))
    llm = LLM(model=str(model), dtype="bfloat16", gpu_memory_utilization=0.8, max_model_len=MODEL_LEN)
    params = SamplingParams(n=1, temperature=0.0, max_tokens=MAX_TOKENS)

    def generate(conversations):
        prompts = [tokenizer(tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False),
                             add_special_tokens=False)["input_ids"] for c in conversations]
        longest = max(len(p) for p in prompts)
        if longest + MAX_TOKENS > MODEL_LEN:
            raise ValueError(f"a judge prompt has {longest} tokens")
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
        return [o.outputs[0].text for o in outputs]

    return generate


def write_bank(path: Path, dataset: str, records: list[dict], judged: list[dict], model: Path) -> dict:
    temporary = path / "llm_judge.jsonl.tmp"
    with temporary.open("w") as handle:
        for record, labels in zip(records, judged, strict=True):
            (reply, correct), (reply_lenient, correct_lenient) = labels["prediction"], labels["prediction_lenient"]
            handle.write(json.dumps({"problem_id": record["problem_id"], "sample_id": record["sample_id"],
                                     "prediction": record["prediction"],
                                     "prediction_lenient": record["prediction_lenient"],
                                     "judge_output": reply, "judge_output_lenient": reply_lenient,
                                     "llm_correct": correct, "llm_correct_lenient": correct_lenient,
                                     "rule_correct": record["rule_correct"]}, ensure_ascii=False) + "\n")
    temporary.replace(path / "llm_judge.jsonl")
    stat = (path / "native.jsonl").stat()
    summary = {"dataset": dataset, "judge_model": str(model), "prompt": "RLCR eval/check_functions.py llm_confidence_verifier",
               "temperature": 0.0, "max_tokens": MAX_TOKENS, "rows": len(records),
               "native_bytes": stat.st_size, "native_mtime_ns": stat.st_mtime_ns,
               "strict_answer_rate": float(np.mean([r["prediction"] is not None for r in records])),
               "lenient_answer_rate": float(np.mean([r["prediction_lenient"] is not None for r in records]))}
    for field, name in zip(FIELDS, ("strict", "lenient"), strict=True):
        llm = [labels[field][1] for labels in judged]
        summary[f"{name}_vs_rule"] = agreement(llm, [r["rule_correct"] for r in records])
        if dataset in CHOICE_DATASETS:
            summary[f"{name}_vs_letter"] = agreement(llm, [r["letter_correct"] for r in records])
    (path / "llm_judge.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:   # refresh the bank's score cache with the new LLM labels (evaluation.score_cache)
        from evaluation.score_cache import load
        load(path, dataset)
    except Exception as error:
        print(f"WARNING score cache {dataset}: {error}", flush=True)
    return summary


def judged_current(path: Path) -> bool:
    try:
        summary = json.loads((path / "llm_judge.json").read_text())
        stat = (path / "native.jsonl").stat()
        return (summary["native_bytes"] == stat.st_size and summary["native_mtime_ns"] == stat.st_mtime_ns
                and (path / "llm_judge.jsonl").is_file())
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--banks", nargs="*", type=Path, default=[],
                        help="bank directories outside run directories (e.g. elicited banks); one dataset only")
    parser.add_argument("--datasets", nargs="+", default=["simpleqa"])
    parser.add_argument("--judge-model", type=Path, default=JUDGE_MODEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="count the queries, load no model")
    args = parser.parse_args()
    generate = None
    for dataset in args.datasets:
        if dataset not in CHOICE_DATASETS | SHORT_ANSWER_DATASETS:
            raise ValueError(f"{dataset}: the LLM judge is for short-answer and multiple-choice sets")
        source = pd.read_parquet(data_path(dataset)).to_dict("records")
        rows_by_problem = {str(r["problem_id"]): r for r in source}
        paths = []
        for path in [bank_path(run, dataset) for run in args.runs] + list(args.banks):
            if not bank_complete(path):
                print(f"SKIP {path}: no complete bank")
            elif judged_current(path) and not args.force:
                print(f"SKIP {path}: judged")
            else:
                paths.append(path)
        if not paths:
            continue
        banks = [read_bank(path, dataset, rows_by_problem) for path in paths]
        if args.dry_run:
            judge_records(dataset, banks, rows_by_problem, lambda queries: ["NO"] * len(queries))
            continue
        if generate is None:
            generate = vllm_generate(args.judge_model)
        judged = judge_records(dataset, banks, rows_by_problem, generate)
        for path, records, labels in zip(paths, banks, judged, strict=True):
            summary = write_bank(path, dataset, records, labels, args.judge_model)
            strict, lenient = summary["strict_vs_rule"], summary["lenient_vs_rule"]
            print(f"{path}: rule acc {strict['other_acc']:.3f} | llm strict acc {strict['llm_acc']:.3f} "
                  f"agreement {strict['agreement']:.3f} | llm lenient acc {lenient['llm_acc']:.3f} "
                  f"agreement {lenient['agreement']:.3f}"
                  + (f" | letter acc {summary['lenient_vs_letter']['other_acc']:.3f} agreement with lenient llm "
                     f"{summary['lenient_vs_letter']['agreement']:.3f}" if "lenient_vs_letter" in summary else ""),
                  flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
