"""Measure the train/rollout mismatch: vLLM sampling vs the HF training forward, on one backbone.

Silent context mismatches (a second BOS token, different RoPE factors, a date in the chat template,
a truncated prompt, a sliding-window cache) all show up the same way: the HF forward that computes the
policy gradient assigns different log-probabilities to the tokens vLLM sampled.  This tool rebuilds
prompts exactly as training does (``tools.response_protocol.chat_messages``, the tokenizer loaded by
``tools.model_protocol.load_chat_tokenizer``, the trainer's prompt tokenisation), samples with vLLM
under the training settings (``max_model_len`` = prompt + completion budget, T, raw log-probabilities),
recomputes every sampled token's raw log-probability with the policy loaded by
``tools.model_protocol.load_policy_model`` in left-padded micro-batches, and reports

* whether the prompt token ids are identical in both paths,
* mean |log p_HF - log p_vLLM| per token, the share of tokens with |delta| > 0.2, and
  the sample estimate of KL(vLLM || HF) per token.

A consistent setup shows bf16 noise only (mean |delta| of order 1e-2 and a KL of order 1e-3).

    CUDA_VISIBLE_DEVICES=0 python -m tools.check_consistency --model resource/models/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch

from tools.model_protocol import load_chat_tokenizer, load_policy_model
from tools.response_protocol import chat_messages
from training.experiment import RESOURCE, resolve



def training_prompt(tokenizer, method: str, problem: str) -> str:
    """The training conversation of third_party/rlcr/dataset_processing.py rendered to text."""
    return tokenizer.apply_chat_template(chat_messages(method, "math", problem), tokenize=False,
                                         add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=lambda p: str(resolve(p)))
    parser.add_argument("--method", default="conftuner-pact")
    parser.add_argument("--prompts", type=int, default=48)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--manifest", default=str(RESOURCE / "datasets/train/prompts-seed-41.parquet"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = load_chat_tokenizer(args.model, padding_side="left")
    rows = pq.read_table(args.manifest).slice(0, args.prompts).to_pylist()
    texts = [training_prompt(tokenizer, args.method, row.get("problem") or row.get("question")) for row in rows]
    # The trainer's prompt tokenisation (third_party/rlcr/GRPO_Trainer.py::_generate_and_score_completions).
    train_ids = [tokenizer(t, add_special_tokens=True)["input_ids"][-args.max_prompt_length:] for t in texts]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_prompt_length + args.max_completion_length, enforce_eager=True, seed=0)
    params = SamplingParams(n=args.samples, temperature=args.temperature, max_tokens=args.max_tokens,
                            logprobs=1, seed=0)
    outputs = llm.generate(texts, params, use_tqdm=False)
    prompt_mismatch = sum(list(o.prompt_token_ids) != ids for o, ids in zip(outputs, train_ids))
    items = []
    for o, ids in zip(outputs, train_ids):
        for c in o.outputs:
            lp = [cand[t].logprob for t, cand in zip(c.token_ids, c.logprobs)]
            items.append((ids, list(c.token_ids), lp))
    del llm
    torch.cuda.empty_cache()

    model = load_policy_model(args.model, torch_dtype=torch.bfloat16,
                              attn_implementation="flash_attention_2").cuda().eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    deltas, kl_terms = [], []
    with torch.no_grad():
        for start in range(0, len(items), args.micro_batch):
            batch = items[start:start + args.micro_batch]
            p_width = max(len(p) for p, _, _ in batch)
            c_width = max(len(c) for _, c, _ in batch)
            ids = torch.full((len(batch), p_width + c_width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, (p, c, _) in enumerate(batch):     # prompts left-padded, completions right-padded
                ids[i, p_width - len(p):p_width] = torch.tensor(p)
                ids[i, p_width:p_width + len(c)] = torch.tensor(c)
                mask[i, p_width - len(p):p_width + len(c)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.float()
            logp = torch.log_softmax(logits[:, p_width - 1:-1], dim=-1)
            for i, (_, c, lp) in enumerate(batch):
                hf = logp[i, torch.arange(len(c)), torch.tensor(c)].cpu()
                v = torch.tensor(lp)
                deltas.append((hf - v).abs())
                kl_terms.append(v - hf)
    d = torch.cat(deltas)
    report = {
        "model": args.model, "method": args.method, "prompts": len(rows), "completions": len(items),
        "tokens": int(d.numel()), "prompt_id_mismatches": int(prompt_mismatch),
        "mean_abs_delta": float(d.mean()), "share_abs_delta_gt_0.2": float((d > 0.2).float().mean()),
        "kl_vllm_hf_per_token": float(torch.cat(kl_terms).mean()),
        "max_prompt_plus_completion": max(len(p) + len(c) for p, c, _ in items),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if not math.isfinite(report["mean_abs_delta"]):
        raise SystemExit("non-finite delta")


if __name__ == "__main__":
    main()
