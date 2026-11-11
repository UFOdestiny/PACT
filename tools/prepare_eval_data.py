"""Build the frozen public evaluation sets under ``$PACT_RESOURCE/datasets/test/``.

Every file has the schema ``problem_id, problem, answer, source``, plus ``choices`` for multiple choice, ``aliases``
for short answers and ``entry_point / code_prefix / code_tests`` for code.  The in-distribution ``pool_heldout`` set
is written by ``tools/build_manifest.py`` from the DeepScaleR pool.  Existing files are kept unless ``--force`` is
given, so the frozen sets stay frozen.

* ``gsm8k`` (``openai/gsm8k`` main, test, 1,319), ``minerva_math`` (``math-ai/minervamath``, 272), ``aime2024``
  (``BytedTsinghua-SIA/AIME-2024``, 30).
* ``amc2024``: the 2024 AMC 12A / 12B problems with an integer answer (34), from the AoPS-wiki scrape
  ``rulins/amc12_22-24``.  Answers that need the answer choices (fractions, radicals, expressions) are dropped.
* ``arc_challenge``: 1,000 ARC-Challenge test questions (``allenai/ai2_arc``), options rendered as ``A. ...``.
* ``simpleqa``: 500 of SimpleQA's 4,326 questions (``basicv8vc/SimpleQA``, test); ``aliases`` is the single gold
  answer, and the paper's labels come from the LLM judge (evaluation/llm_judge.py).
* ``mbpp_plus_fenced``: the MBPP+ problems and tests (``evalplus/mbppplus``, test) in the HumanEval+ style: the user
  message asks for the complete function and shows a ```python``` stub (the signature of the reference function the
  first base assert calls, with a docstring holding the task and the three base asserts); ``code_prefix`` is the
  test imports plus that stub.  Dropped: MBPP 590 (its reference solution fails its own tests) and 737 / 787 / 794
  (their tests never assert, so the bare stub would pass); 374 remain.

``DECONTAMINATION`` builds seven more benchmarks that no experiment evaluates: the training pool is decontaminated
against them as well (tools/prepare_train_pool.py), which reads only their problem text.  The texts match the frozen
files except for MMLU-Pro, whose frozen 2,000-question sample came from an earlier sampling of the same split; 29 of
the pool's 598 contamination exclusions matched it, and every exclusion is listed in ``provenance/pool_decontamination.jsonl``.

    python -m tools.prepare_eval_data [--only gsm8k simpleqa ...] [--force]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

from training.experiment import RESOURCE

TEST = RESOURCE / "datasets/test"
EXCLUDED = ("mbpp-590", "mbpp-737", "mbpp-787", "mbpp-794")


def render_choices(question: str, options: list[str]) -> str:
    lines = "\n".join(f"{chr(ord('A') + i)}. {text}" for i, text in enumerate(options))
    return f"{question.strip()}\n\nOptions:\n{lines}"


def sample_indices(size: int, limit: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(size), min(limit, size)))


def stratified(rows_by_group: dict[str, list], target: int, seed: int = 82) -> list:
    """Largest-remainder allocation of ``target`` rows across groups, fixed seed."""
    total = sum(len(v) for v in rows_by_group.values())
    exact = {g: target * len(v) / total for g, v in rows_by_group.items()}
    counts = {g: int(x) for g, x in exact.items()}
    for g in sorted(rows_by_group, key=lambda g: -(exact[g] - counts[g]))[: target - sum(counts.values())]:
        counts[g] += 1
    rng = random.Random(seed)
    out = []
    for g in sorted(rows_by_group):
        rows = sorted(rows_by_group[g], key=lambda r: r["problem_id"])
        out.extend(rng.sample(rows, counts[g]))
    return out


# --- mathematics ---------------------------------------------------------------

def build_gsm8k() -> pd.DataFrame:
    data = load_dataset("openai/gsm8k", "main", split="test")
    return pd.DataFrame([{"problem_id": f"gsm8k-test-{i:04d}", "problem": r["question"],
                          "answer": str(r["answer"]).split("####")[-1].strip(), "source": "gsm8k"}
                         for i, r in enumerate(data)])


def build_minerva_math() -> pd.DataFrame:
    data = load_dataset("math-ai/minervamath", split="test")
    return pd.DataFrame([{"problem_id": f"minerva-{i:04d}", "problem": r["question"], "answer": str(r["answer"]),
                          "source": "minerva_math"} for i, r in enumerate(data)])


def build_aime2024() -> pd.DataFrame:
    """The 30 AIME 2024 problems in their first-occurrence order (the source repeats each problem 32 times)."""
    data = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")
    rows, seen = [], set()
    for r in data:
        index = int(r["extra_info"]["index"])
        if index in seen:
            continue
        seen.add(index)
        rows.append({"problem_id": f"aime2024-{index}", "problem": str(r["extra_info"]["raw_problem"]),
                     "answer": str(r["reward_model"]["ground_truth"]), "source": "aime2024"})
    return pd.DataFrame(rows)


def build_amc2024() -> pd.DataFrame:
    data = load_dataset("rulins/amc12_22-24", split="train")
    rows = []
    for r in data:
        match = re.search(r"2024_AMC_12([AB])_Problems/Problem_(\d+)", str(r["url"]))
        answer = str(r["answer"]).replace(",", "").strip()
        if match and re.fullmatch(r"-?\d+", answer):
            rows.append({"problem_id": f"amc2024-12{match.group(1)}-{int(match.group(2)):02d}",
                         "problem": str(r["problem"]).strip(), "answer": answer, "source": "amc2024"})
    return pd.DataFrame(sorted(rows, key=lambda row: row["problem_id"]))


# --- out of distribution ---------------------------------------------------------

def build_arc_challenge() -> pd.DataFrame:
    data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for index in sample_indices(len(data), 1000, seed=82):
        item = data[index]
        labels = list(item["choices"]["label"])
        texts = [str(t).strip() for t in item["choices"]["text"]]
        if str(item["answerKey"]) not in labels:
            continue
        gold = labels.index(str(item["answerKey"]))
        rows.append({"problem_id": f"arc-challenge-{item['id']}", "problem": render_choices(str(item["question"]), texts),
                     "answer": chr(ord("A") + gold), "choices": texts, "category": "science", "source": "arc_challenge"})
    return pd.DataFrame(rows)


def build_simpleqa() -> pd.DataFrame:
    data = load_dataset("basicv8vc/SimpleQA", split="test")
    rows = []
    for index in sample_indices(len(data), 500, seed=0):
        r = data[index]
        metadata = ast.literal_eval(str(r["metadata"]))   # a dict literal, never evaluated as code
        answer = str(r["answer"]).strip()
        rows.append({"problem_id": f"simpleqa-{index:04d}",
                     "problem": f"Answer the question with a short factual answer.\n\n{str(r['problem']).strip()}",
                     "answer": answer, "aliases": [answer], "category": str(metadata.get("topic", "")),
                     "source": "simpleqa"})
    return pd.DataFrame(rows)


def mbpp_stub(code: str, description: str, asserts: list[str]) -> tuple[str, str]:
    """``(entry_point, stub)``: the reference function that the first base assert calls, as ``def name(args):`` with
    a docstring holding the description and the asserts (raw, or in the other quotes, when the text needs it)."""
    functions = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)]
    called = [node.func.id for node in ast.walk(ast.parse(asserts[0]))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    matches = [f for f in functions if f.name in called]
    if len(matches) != 1:
        raise ValueError(f"no unique reference function for {asserts[0]!r}")
    function = matches[0]
    text = "\n".join([description, "Your code should pass these tests:", *asserts])
    quote = "'" * 3 if '"' * 3 in text else '"' * 3
    raw = "r" if "\\" in text else ""
    body = "\n".join(f"    {line}" if line.strip() else "" for line in text.splitlines())
    returns = f" -> {ast.unparse(function.returns)}" if function.returns is not None else ""
    stub = f"def {function.name}({ast.unparse(function.args)}){returns}:\n    {raw}{quote}\n{body}\n    {quote}\n"
    ast.parse(stub)
    return function.name, stub


def build_mbpp_plus_fenced() -> pd.DataFrame:
    data = load_dataset("evalplus/mbppplus", split="test")
    rows = []
    for r in data:
        if f"mbpp-{int(r['task_id']):03d}" in EXCLUDED:
            continue
        entry_point, stub = mbpp_stub(str(r["code"]), str(r["prompt"]).strip(), [str(t) for t in r["test_list"]])
        imports = "\n".join(str(i) for i in r["test_imports"])
        rows.append({"problem_id": f"mbpp-{int(r['task_id']):03d}",
                     "problem": "Complete the following Python function. Give the complete function, including its "
                                f"signature and any imports it needs.\n\n```python\n{stub.strip()}\n```",
                     "answer": str(r["code"]), "entry_point": entry_point,
                     "code_prefix": f"{imports}\n\n{stub}" if imports else stub, "code_tests": str(r["test"]),
                     "category": "code", "source": "mbpp_plus_fenced"})
    return pd.DataFrame(rows)


# --- decontamination only ----------------------------------------------------------

def build_math500() -> pd.DataFrame:
    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return pd.DataFrame([{"problem_id": str(r["unique_id"]), "problem": r["problem"], "answer": str(r["answer"]),
                          "source": "math500"} for r in data])


def build_olympiad_bench() -> pd.DataFrame:
    """Text-only, final-answer English subset (OE_TO_maths_en_COMP)."""
    data = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    rows = []
    for r in data:
        answers = r.get("final_answer") or []
        if len(answers) != 1:
            continue
        rows.append({"problem_id": f"olympiad-{r['id']}", "problem": str(r["question"]),
                     "answer": str(answers[0]).strip().strip("$"), "source": "olympiad_bench"})
    return pd.DataFrame(rows)


def build_aime2025() -> pd.DataFrame:
    data = load_dataset("PrimeIntellect/AIME-25", split="train")
    return pd.DataFrame([{"problem_id": str(r["problem_id"]),
                          "problem": re.sub(r"\nMark your solution with \\boxed\.$", "",
                                            str(r["prompt"]).removeprefix("Problem: ")),
                          "answer": str(json.loads(r["verification_info"])["ground_truth"]),
                          "source": "aime2025"} for r in data])


def build_amc2023() -> pd.DataFrame:
    data = load_dataset("math-ai/amc23", split="test")
    return pd.DataFrame([{"problem_id": f"amc2023-{r['id']}", "problem": str(r["question"]),
                          "answer": str(float(r["answer"])), "source": "amc2023"} for r in data])


def build_mmlu_pro() -> pd.DataFrame:
    data = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    groups = defaultdict(list)
    for r in data:
        choices = [str(c).strip() for c in r["options"]]
        groups[str(r["category"])].append({
            "problem_id": f"mmlu-pro-{r['question_id']}", "question": str(r["question"]).strip(),
            "problem": render_choices(str(r["question"]), choices), "answer": str(r["answer"]).strip().upper(),
            "choices": choices, "category": str(r["category"]), "source": "mmlu_pro"})
    return pd.DataFrame(stratified(groups, 2000))


def build_trivia_qa() -> pd.DataFrame:
    data = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows, seen = [], set()
    for index in sample_indices(len(data), 1000, seed=82):
        item = data[index]
        if str(item["question_id"]) in seen:
            continue
        seen.add(str(item["question_id"]))
        answer = item["answer"]
        aliases = sorted({str(a).strip() for a in answer.get("aliases", []) if str(a).strip()})
        rows.append({"problem_id": f"trivia-{len(rows):04d}-{item['question_id']}",
                     "problem": f"Answer the question with a short factual answer.\n\n{str(item['question']).strip()}",
                     "answer": str(answer["value"]).strip(), "aliases": aliases or [str(answer["value"]).strip()],
                     "category": "trivia", "source": "trivia_qa"})
    return pd.DataFrame(rows)


def build_supergpqa() -> pd.DataFrame:
    data = load_dataset("m-a-p/SuperGPQA", split="train")
    groups = defaultdict(list)
    for r in data:
        choices = [str(c).strip() for c in r["options"]]
        if len(choices) > 10:
            continue
        groups[str(r["discipline"])].append({
            "problem_id": f"supergpqa-{r['uuid']}", "problem": render_choices(str(r["question"]), choices),
            "answer": str(r["answer_letter"]).strip().upper(), "choices": choices,
            "category": str(r["discipline"]), "source": "supergpqa"})
    return pd.DataFrame(stratified(groups, 2000))


DECONTAMINATION = {"math500": build_math500, "olympiad_bench": build_olympiad_bench, "aime2025": build_aime2025,
                   "amc2023": build_amc2023, "mmlu_pro": build_mmlu_pro, "trivia_qa": build_trivia_qa,
                   "supergpqa": build_supergpqa}

BUILDERS = {"gsm8k": build_gsm8k, "minerva_math": build_minerva_math, "aime2024": build_aime2024,
            "amc2024": build_amc2024, "arc_challenge": build_arc_challenge, "simpleqa": build_simpleqa,
            "mbpp_plus_fenced": build_mbpp_plus_fenced, **DECONTAMINATION}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(BUILDERS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    TEST.mkdir(parents=True, exist_ok=True)
    for name, build in BUILDERS.items():
        if args.only and name not in args.only:
            continue
        path = TEST / f"{name}.parquet"
        if path.exists() and not args.force:
            print(f"KEEP {path}")
            continue
        frame = build()
        ids = sorted(frame["problem_id"])
        if any(a == b for a, b in zip(ids, ids[1:])):
            raise ValueError(f"{name}: duplicate problem IDs")
        frame.to_parquet(path, index=False)
        answers = frame["answer"].astype(str)
        print(f"WROTE {path} ({len(frame)} problems; answers "
              f"{json.dumps(answers.value_counts().head(4).to_dict()) if name == 'arc_challenge' else 'free-form'})")


if __name__ == "__main__":
    main()
