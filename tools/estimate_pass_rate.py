"""Label every pool problem with Mathstral's pass rate (vLLM, K stochastic samples).

The shared dataset is labelled once with ``mistralai/Mathstral-7B-v0.1`` under
its native chat template and a neutral boxed-answer instruction (K=8, T=0.7).
``tools/build_manifest.py`` then constructs the shared train / dev / test
splits.  A different backbone may be run later on the frozen 8,000-problem
training manifest for diagnostics, but never changes split membership.

Output: ``$PACT_RESOURCE/datasets/train/passrate-<backbone>-screening.parquet`` with columns
``problem_id, pass_rate, correct, samples, mean_completion_tokens`` and the
raw per-sample verdicts in a sibling ``.jsonl`` for auditing.

    CUDA_VISIBLE_DEVICES=0 python -m tools.estimate_pass_rate \
        --model resource/models/Mathstral-7B-v0.1 --backbone mathstral-7b-v0.1 --samples 8

With more than one GPU, run one strided shard per GPU and merge them; the shard
files are deleted by the merge, and a finished shard is never regenerated:

    for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python -m tools.estimate_pass_rate \
        --model ... --backbone ... --shards 4 --shard $k & done; wait
    python -m tools.estimate_pass_rate --backbone ... --shards 4 --merge
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.generate import parse_answer, required_model_len
from evaluation import tasks
from tools.model_protocol import load_chat_tokenizer
from tools.response_protocol import system_prompt
from training.experiment import ROOT, RESOURCE, resolve

DATASETS = RESOURCE / "datasets"


def judge_answer(item: tuple[str | None, str]) -> bool:
    """Verdict for one raw ``<answer>`` string against its gold (pool rows are all math)."""
    raw, gold = item
    prediction = None if raw is None else tasks.parse("math", raw, None)
    return bool(tasks.judge("math", prediction, gold, None))


def merge(stem: str, shards: int, samples: int, pool_path: Path) -> None:
    """Concatenate the shard outputs into ``<stem>.parquet``/``.jsonl`` and drop the shards."""
    out_dir = DATASETS / "train"
    parts = [out_dir / f"{stem}-shard-{k}-of-{shards}.parquet" for k in range(shards)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"unfinished shards: {', '.join(missing)}")
    table_rows, verdict_lines, wall = [], [], 0.0
    reports = []
    for k, part in enumerate(parts):
        table_rows.extend(pq.read_table(part).to_pylist())
        verdict_lines.extend(part.with_suffix(".jsonl").read_text().splitlines())
        report = json.loads((DATASETS / "provenance" / f"{stem}-shard-{k}-of-{shards}.json").read_text())
        reports.append(report)
        wall = max(wall, float(report["wall_seconds"]))
    invariant_keys = ("backbone", "model", "pool", "samples", "temperature", "seed",
                      "response_protocol", "system_prompt", "max_tokens")
    for key in invariant_keys:
        values = {json.dumps(report.get(key), sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"shard reports disagree on {key}: {values}")
    order = {str(pid): i for i, pid in
             enumerate(pq.read_table(pool_path, columns=["problem_id"])
                       .column("problem_id").to_pylist())}
    ids = [str(row["problem_id"]) for row in table_rows]
    if len(ids) != len(order) or len(set(ids)) != len(ids) or set(ids) != set(order):
        raise ValueError("merged shards do not cover the pool exactly once")
    table_rows.sort(key=lambda r: order[str(r["problem_id"])])
    verdict_lines.sort(key=lambda line: order[str(json.loads(line)["problem_id"])])
    pq.write_table(pa.Table.from_pylist(table_rows), out_dir / f"{stem}.parquet")
    (out_dir / f"{stem}.jsonl").write_text("\n".join(verdict_lines) + "\n")
    histogram = [0] * (samples + 1)
    for r in table_rows:
        histogram[int(r["correct"])] += 1
    summary = {key: reports[0][key] for key in invariant_keys}
    summary.update({"stem": stem, "shards": shards, "problems": len(table_rows),
                    "slowest_shard_seconds": wall, "correct_histogram": histogram,
                    "mean_pass_rate": sum(r["pass_rate"] for r in table_rows) / len(table_rows)})
    summary["shard_wall_seconds"] = [float(report["wall_seconds"]) for report in reports]
    (DATASETS / "provenance" / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, part in enumerate(parts):
        part.unlink()
        part.with_suffix(".jsonl").unlink()
        (DATASETS / "provenance" / f"{stem}-shard-{k}-of-{shards}.json").unlink()
    print(json.dumps(summary, indent=2))


def read_progress(path: Path, samples: int) -> tuple[dict[str, dict], list[str]]:
    """Return complete JSONL records and canonical lines, dropping a torn tail."""
    records: dict[str, dict] = {}
    lines: list[str] = []
    if not path.is_file():
        return records, lines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            problem_id = str(record["problem_id"])
            if int(record["samples"]) != samples or len(record["responses"]) != samples:
                raise ValueError("wrong sample count")
            if problem_id in records:
                raise ValueError("duplicate problem")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            break
        records[problem_id] = record
        lines.append(json.dumps(record, ensure_ascii=False))
    return records, lines


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, help="artifact name of the backbone, e.g. llama-3.1-8b-instruct")
    parser.add_argument("--pool", type=Path, default=DATASETS / "train/pool.parquet")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--limit", type=int, help="only the first N pool rows (smoke test)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--shards", type=int, default=1, help="number of strided shards (one GPU each)")
    parser.add_argument("--shard", type=int, default=0, help="index of this shard")
    parser.add_argument("--merge", action="store_true", help="combine finished shards into the final files")
    parser.add_argument("--judge-workers", type=int, default=8, help="processes used for the judging pass")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="prompts generated and durably written per resumable batch")
    parser.add_argument("--model", dest="model", type=resolve)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    stem = f"passrate-{args.backbone}-screening" + (f"-first{args.limit}" if args.limit else "")
    pool_path = args.pool if args.pool.is_absolute() else ROOT / args.pool
    if args.merge:
        merge(stem, args.shards, args.samples, pool_path)
        return
    if args.model is None:
        parser.error("--model is required unless --merge is given")
    from vllm import LLM, SamplingParams

    rows = [{**row, "_pool_index": index}
            for index, row in enumerate(pq.read_table(pool_path).to_pylist())]
    if args.limit:
        rows = rows[: args.limit]
    if args.shards > 1:
        # Strided so every shard sees the same mix of sources and lengths.
        rows = rows[args.shard::args.shards]
        stem = f"{stem}-shard-{args.shard}-of-{args.shards}"
        if (DATASETS / "train" / f"{stem}.parquet").is_file():
            print(f"SKIP {stem}: already finished")
            return
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    tokenizer = load_chat_tokenizer(args.model)
    screening_prompt = system_prompt("screening", "math")
    prompts = {
        str(row["problem_id"]): tokenizer.apply_chat_template(
            [{"role": "system", "content": screening_prompt},
             {"role": "user", "content": f"\n\nPROBLEM: {row['problem']}\n\n"}],
            add_generation_prompt=True, tokenize=False,
        )
        for row in rows
    }
    model_len = required_model_len([len(tokenizer.encode(p)) for p in prompts.values()], args.max_tokens)
    llm = LLM(model=str(args.model), dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, max_model_len=model_len)
    started = time.time()
    out_dir = DATASETS / "train"
    raw_path = out_dir / f"{stem}.jsonl"
    progress, canonical_lines = read_progress(raw_path, args.samples)
    if raw_path.is_file() and len(canonical_lines) != len(raw_path.read_text(encoding="utf-8").splitlines()):
        temporary = raw_path.with_suffix(".jsonl.tmp")
        temporary.write_text("\n".join(canonical_lines) + ("\n" if canonical_lines else ""), encoding="utf-8")
        temporary.replace(raw_path)
    pending = [row for row in rows if str(row["problem_id"]) not in progress]
    generation_seconds = 0.0
    judge_seconds = 0.0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        batch_prompts = [prompts[str(row["problem_id"])] for row in batch]
        params = [SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                                 max_tokens=args.max_tokens, seed=args.seed + int(row["_pool_index"]))
                  for row in batch]
        generation_started = time.time()
        outputs = llm.generate(batch_prompts, params)
        generation_seconds += time.time() - generation_started
        answers = [[parse_answer(output.text) for output in request.outputs] for request in outputs]
        work = [(answer, str(row["answer"]))
                for row, group in zip(batch, answers, strict=True) for answer in group]
        judge_started = time.time()
        if args.judge_workers > 1:
            with ProcessPoolExecutor(max_workers=args.judge_workers) as workers:
                flat = list(workers.map(judge_answer, work, chunksize=256))
        else:
            flat = [judge_answer(item) for item in work]
        judge_seconds += time.time() - judge_started
        batch_records, position = [], 0
        for row, group, request in zip(batch, answers, outputs, strict=True):
            verdicts = flat[position:position + len(group)]
            position += len(group)
            responses = [
                {"answer": answer, "correct": bool(verdict), "text": output.text,
                 "finish_reason": output.finish_reason, "completion_tokens": len(output.token_ids)}
                for answer, verdict, output in zip(group, verdicts, request.outputs, strict=True)
            ]
            batch_records.append({
                "problem_id": str(row["problem_id"]), "correct": int(sum(verdicts)),
                "samples": len(verdicts), "pass_rate": sum(verdicts) / len(verdicts),
                "mean_completion_tokens": sum(item["completion_tokens"] for item in responses) / len(responses),
                "responses": responses,
            })
        append_records(raw_path, batch_records)
        progress.update({record["problem_id"]: record for record in batch_records})

    if set(progress) != {str(row["problem_id"]) for row in rows}:
        raise RuntimeError("shard progress does not cover every assigned problem")
    table_rows = [{key: record[key] for key in
                   ("problem_id", "pass_rate", "correct", "samples", "mean_completion_tokens")}
                  for record in (progress[str(row["problem_id"])] for row in rows)]
    pq.write_table(pa.Table.from_pylist(table_rows), out_dir / f"{stem}.parquet")
    histogram = [0] * (args.samples + 1)
    for r in table_rows:
        histogram[r["correct"]] += 1
    summary = {"backbone": args.backbone, "model": str(args.model),
               "pool": str(pool_path.relative_to(ROOT)) if pool_path.is_relative_to(ROOT) else str(pool_path),
               "problems": len(table_rows), "samples": args.samples, "temperature": args.temperature,
               "seed": args.seed, "shard": args.shard, "shards": args.shards,
               "response_protocol": "screening",
               "system_prompt": screening_prompt, "max_tokens": args.max_tokens,
               "batch_size": args.batch_size, "resumed_problems": len(rows) - len(pending),
               "wall_seconds": time.time() - started, "generation_seconds": generation_seconds,
               "judge_seconds": judge_seconds, "judge_workers": args.judge_workers,
               "correct_histogram": histogram,
               "mean_pass_rate": sum(r["pass_rate"] for r in table_rows) / len(table_rows)}
    (DATASETS / "provenance" / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
