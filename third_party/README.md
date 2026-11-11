# Training dependencies

`rlcr/` contains the training modules of `damanimehul/RLCR` at commit
`be2624b1753d738b9552df4dd6d50ac33b3668ee`, with local changes: parquet input,
Transformers 5 / vLLM 0.19 support, completion-logprob alignment, backbone loading
through `tools/model_protocol.py`, prompts built by
`tools.response_protocol.chat_messages` and tokenised like vLLM
(`add_special_tokens=True`), completion masks taken from vLLM's completion lengths
and finish reasons (so turn terminators other than `eos_token` are handled),
frozen parameters skipped when weights are pushed to vLLM, sequence- or
token-level advantages supplied by the shared trainer (`broadcast_advantages`),
an optional clip of the k3 KL estimator's log ratio, and checkpoint metrics.
Its MIT license is retained in `rlcr/LICENSE`.

TRL is installed into `trl/` at commit `69ad852e5654a77f1695eb4c608906fe0c7e8624`
with the import adjustment in `patches/trl-transformers5.patch`; run
`python -m tools.setup_trl` to fetch it. The checkout is ignored by Git.

`training/trainer.py` subclasses `rlcr/GRPO_Trainer.py`, and
`training/entrypoint.py` runs the training loop of `rlcr/rl_runner.py` with that
trainer and the judge-backed rewards of `training/rewards.py`.
