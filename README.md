# PACT: Peers As Calibration Tests

Code for *Peers as Tests, Not Targets: Calibrating LLM Confidence under Difficulty Shift*.

PACT trains a language model to answer and to state a verbalized confidence (`Confidence: 72%`) that stays calibrated
when the mix of easy and hard questions shifts. Group-based RL already samples several responses per question and
verifies each one. PACT keeps each response's own correctness as its confidence target and uses the other responses to
the same question (its *peers*) only as tests: responses are grouped into strata by their leave-one-out peer pass rate,
and a centered penalty removes differences in mean confidence error across strata. The tests are added to ConfTuner's
expected Brier objective; training reuses the RL responses, and deployment needs one response and no verifier.

## Setup

Python 3.11 and CUDA GPUs (each run in the paper used 4 GPUs).

```bash
pip install -r requirements.txt
python -m tools.setup_trl            # pinned TRL checkout in third_party/trl
cp .env.example .env                 # local paths: PACT_RESOURCE, PACT_ARTIFACTS, HF_HOME, HF_TOKEN
```

Models go in `$PACT_RESOURCE/models/` (`Llama-3.1-8B-Instruct`, `olmo3-8B`, and `Mathstral-7B-v0.1`, which only
annotates training-question difficulty).

## Data

```bash
python -m tools.prepare_eval_data                      # evaluation and decontamination sets
python -m tools.prepare_train_pool                     # DeepScaleR pool, decontaminated
CUDA_VISIBLE_DEVICES=0 python -m tools.estimate_pass_rate \
    --model resource/models/Mathstral-7B-v0.1 --backbone mathstral-7b-v0.1 --samples 8
python -m tools.build_manifest                         # 8,000 train / 2,000 held-out questions
```

Evaluation: ID mathematics (held-out pool, GSM8K, Minerva, AIME 2024, AMC 2024) and OOD (ARC-Challenge, SimpleQA,
MBPP+).

## Training and evaluation

```bash
bash scripts/run.sh configs/llama-3.1-8b-instruct/conftuner-pact.yaml 41 0,1,2,3
```

This trains 125 updates of 64 questions × 16 responses, samples 16 responses per test question, and judges them.
Outputs go to `$PACT_ARTIFACTS/<model>/<method>/seed-<n>/`, with a score cache (`scores.npz`, `summary.json`) per
dataset. Completed stages are skipped on re-runs.

| Config | Paper name |
|---|---|
| `conftuner-pact` | ConfTuner+PACT (seeds 41–43) |
| `peer-target` | Peer-Target (peers mixed into the target) |
| `pact-agree`, `pact-self`, `pact-extdiff`, `pact-shuffled` | PACT-Agree, PACT-Self, PACT-ExtDiff, PACT-Shuffled |
| `pact-b3`, `pact-b16`, `pact-g4` | B = 3, B = 16, γ = 4 (Llama only) |

## Repository

```
method/        PACT (pact.py) and the ConfTuner credit it extends (conftuner.py)
training/      method registry, shared GRPO trainer, exact grid Brier loss, launcher (run.py)
evaluation/    sampling, judges (math, lenient multiple choice, LLM judge, sandboxed code), metrics
tools/         data preparation, model/prompt protocol, consistency check, checkpoint cleanup
configs/       shared training settings and one YAML per experiment
scripts/       run.sh
third_party/   vendored RLCR trainer and the TRL patch
```

## License

MIT ([LICENSE](LICENSE)). Third-party code, models and datasets keep their own licenses.
