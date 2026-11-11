"""Build the candidate training pool from DeepScaleR-Preview-Dataset.

Writes ``resource/datasets/train/pool.parquet`` with the schema
``problem_id, problem, answer, source, origin`` after

1. loading ``agentica-org/DeepScaleR-Preview-Dataset`` (40,315 rows; AIME
   1984-2023, AMC before 2023, Omni-MATH, Still) and de-duplicating by
   normalised problem text, dropping duplicate groups whose answers disagree;
2. removing every problem that is contaminated against an evaluation file under ``$PACT_RESOURCE/datasets/test/``
   (``DECONTAMINATION_SETS``, built by tools/prepare_eval_data.py): an exact normalised-text match, an exact
   match after all digits are removed (a renumbered copy), or a shared
   13-word window (the n-gram rule of GPT-3 / Light-R1), on either side;
3. keeping only problems whose gold answer the judge can mark correct against
   itself, so no unsolvable row enters the pool.

The pool carries no difficulty label: ``tools/estimate_pass_rate.py`` labels it
once with the protocol's reference labeler and ``tools/build_manifest.py`` cuts
the shared train / test / dev splits from those labels.  The dropped rows and
the reason for each are recorded in ``provenance/pool_decontamination.jsonl``.

    python -m tools.prepare_train_pool
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation import tasks
from training.experiment import ROOT, RESOURCE

DATASETS = RESOURCE / "datasets"
DEEPSCALER_REPO = "agentica-org/DeepScaleR-Preview-Dataset"
NGRAM = 13
# The benchmarks the pool is decontaminated against (the evaluation sets that existed when the pool was frozen).
DECONTAMINATION_SETS = ("aime2024", "aime2025", "amc2023", "arc_challenge", "gsm8k", "math500", "minerva_math",
                        "mmlu_pro", "olympiad_bench", "supergpqa", "trivia_qa")
_WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text))).casefold()


def digitless(text: str) -> str:
    return re.sub(r"\d", "", normalise(text))


def word_ngrams(text: str, n: int = NGRAM) -> set[str]:
    words = _WORD.findall(unicodedata.normalize("NFKC", str(text)).casefold())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_deepscaler() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DEEPSCALER_REPO, "deepscaler.json", repo_type="dataset")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"problem": str(r["problem"]).strip(), "answer": str(r["answer"]).strip(), "source": "deepscaler",
             "origin": f"deepscaler:{index}"} for index, r in enumerate(rows)]


class Contamination:
    """Exact, digit-stripped and 13-word-window matches against the evaluation files."""

    def __init__(self, folder: Path) -> None:
        self.exact: dict[str, str] = {}
        self.digitless: dict[str, str] = {}
        self.ngrams: dict[str, str] = {}
        for name in DECONTAMINATION_SETS:
            path = folder / f"{name}.parquet"
            table = pq.read_table(path)
            column = "question" if "question" in table.schema.names else "problem"
            for text in table.column(column).to_pylist():
                self.add(str(text), path.stem)

    def add(self, text: str, benchmark: str) -> None:
        self.exact.setdefault(normalise(text), benchmark)
        self.digitless.setdefault(digitless(text), benchmark)
        for gram in word_ngrams(text):
            self.ngrams.setdefault(gram, benchmark)

    def match(self, problem: str) -> tuple[str, str] | None:
        key = normalise(problem)
        if key in self.exact:
            return "exact", self.exact[key]
        if digitless(problem) in self.digitless:
            return "digitless", self.digitless[digitless(problem)]
        for gram in word_ngrams(problem):
            if gram in self.ngrams:
                return "ngram", self.ngrams[gram]
        return None


def build(rows: list[dict], contamination: Contamination) -> tuple[list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalise(row["problem"])].append(row)
    kept, dropped, report = [], [], defaultdict(int)
    report["rows"] = len(rows)
    report["unique_problems"] = len(groups)
    for members in groups.values():
        row = members[0]
        report["duplicate_rows"] += len(members) - 1
        answers = {normalise(m["answer"]) for m in members}
        if len(answers) > 1:
            report["conflicting_answer_groups"] += 1
            dropped.append({**row, "reason": "conflicting_answers", "answers": sorted(answers)})
            continue
        hit = contamination.match(row["problem"])
        if hit is not None:
            report[f"contaminated:{hit[0]}"] += 1
            dropped.append({**row, "reason": f"contaminated:{hit[0]}", "benchmark": hit[1]})
            continue
        if not row["answer"] or not tasks.judge("math", row["answer"], row["answer"], None):
            report["unjudgeable_gold"] += 1
            dropped.append({**row, "reason": "unjudgeable_gold"})
            continue
        kept.append({"problem_id": "pool-" + row["origin"].replace(":", "-"), **row})
    report["kept"] = len(kept)
    return kept, dropped, dict(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATASETS / "train/pool.parquet")
    args = parser.parse_args()
    contamination = Contamination(DATASETS / "test")
    kept, dropped, report = build(load_deepscaler(), contamination)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(kept), temporary)
    temporary.replace(args.out)
    provenance = DATASETS / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    with (provenance / "pool_decontamination.jsonl").open("w", encoding="utf-8") as handle:
        for row in dropped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"repo": DEEPSCALER_REPO, "output": str(args.out.relative_to(ROOT)), "ngram_words": NGRAM,
               "evaluation_files": sorted(DECONTAMINATION_SETS), **dict(sorted(report.items()))}
    (provenance / "pool.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
