"""One training process: parse the resolved configuration and train ``ExperimentTrainer``.

``training/run.py`` launches this file once per GPU through ``accelerate`` with ``--config <run>/training/resolved.yaml``.
The run loop follows the vendored ``third_party/rlcr/rl_runner.py`` (its argument classes, logging, prompt rendering
and model kwargs), with this repository's trainer and judge-backed reward functions (``training/rewards.py``).
"""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Launched by path: the vendored RLCR modules, the pinned TRL checkout and the repository come first.
sys.path[:0] = [str(ROOT / "third_party/rlcr"), str(ROOT / "third_party/trl"), str(ROOT)]

import datasets  # noqa: E402
from arguments import GRPOConfig, GRPOScriptArguments, ModelConfig  # noqa: E402
from dataset_processing import process_dataset  # noqa: E402
from rl_runner import logger_setup, model_init  # noqa: E402
from transformers import set_seed  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import TrlParser  # noqa: E402

from training.rewards import (accuracy_reward, brier_reward, confidence_one_or_zero, format_reward,  # noqa: E402
                              mean_confidence_reward)
from training.trainer import ExperimentTrainer  # noqa: E402
from training.trainer_metrics import MetricsHistoryCallback  # noqa: E402

logger = logging.getLogger(__name__)


def reward_functions(script_args) -> list:
    """The configured reward functions, in ``reward_funcs`` order."""
    registry = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "accuracy": partial(accuracy_reward, format_pattern=script_args.format_pattern),
        "brier": partial(brier_reward, format_pattern=script_args.format_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    return [registry[name] for name in script_args.reward_funcs]


def main(script_args, training_args, model_args) -> None:
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args)
    logger.setLevel(training_args.get_process_log_level())
    last_checkpoint = (get_last_checkpoint(training_args.output_dir)
                       if os.path.isdir(training_args.output_dir) else None)
    if script_args.data_train_file is None:
        raise ValueError("data_train_file is required")
    dataset = datasets.DatasetDict({"train": datasets.load_dataset("parquet", data_files=script_args.data_train_file,
                                                                    split="train")})
    dataset = process_dataset(dataset, script_args)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    training_args.model_init_kwargs = model_init(model_args, training_args)
    train_dataset = dataset["train"]

    trainer = ExperimentTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_functions(script_args),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[MetricsHistoryCallback()],
    )
    logger.info("*** Train ***")
    checkpoint = (training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None
                  else last_checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["rl-verify"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main(*TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig)).parse_args_and_config())
