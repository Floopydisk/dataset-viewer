# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""Prefix Tuning: prepend trainable continuous prefix tokens to each transformer layer."""

import logging
from typing import Any, Optional

from datasets import Dataset
from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments

from worker.training._base import get_device, log_epoch, resolve_output_dir, resolve_task, set_seed
from worker.training._data import load_splits, resolve_seq2seq_columns, resolve_text_column
from worker.training.algorithms import TrainingAlgorithmResult, TrainingExecutionContext

_NUM_VIRTUAL_TOKENS = 20
# PrefixTuningConfig has reliable PEFT support only for generative task types
_SUPPORTED_TASK_TYPES = frozenset(["seq2seq", "causal-lm", "text-classification"])


def _load_prefix_model(model_name: str, task_type: str) -> tuple[Any, int]:
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Prefix tuning does not support task_type '{task_type}'. "
            f"Supported: {sorted(_SUPPORTED_TASK_TYPES)}"
        )
    model_class, peft_task_type = resolve_task(task_type)
    base: PreTrainedModel = model_class.from_pretrained(model_name)
    config = PrefixTuningConfig(task_type=peft_task_type, num_virtual_tokens=_NUM_VIRTUAL_TOKENS)
    model = get_peft_model(base, config)
    model.to(get_device())
    trainable, total = model.get_nb_trainable_parameters()
    logging.info(f"Prefix tuning ({task_type}): {trainable:,} / {total:,} parameters trainable")
    return model, trainable


def _tokenize_split(ds: Dataset, tokenizer: Any, task_type: str) -> Dataset:
    if task_type == "seq2seq":
        input_col, target_col = resolve_seq2seq_columns(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            enc = tokenizer(  # type: ignore[operator]
                batch[input_col], truncation=True, padding="max_length", max_length=128
            )
            with tokenizer.as_target_tokenizer():
                enc["labels"] = tokenizer(  # type: ignore[operator]
                    batch[target_col], truncation=True, padding="max_length", max_length=128
                )["input_ids"]
            return enc
    else:
        col = resolve_text_column(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(batch[col], truncation=True, padding="max_length", max_length=128)  # type: ignore[operator]

    return ds.map(_tok, batched=True)


def _build_trainer(
    model: Any,
    train_ds: Dataset,
    eval_ds: Optional[Dataset],
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Trainer:
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        logging_steps=50,
        report_to="none",
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)


def run(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    set_seed(context["seed"])

    model_name = context["model_name"]
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    task_type = context["task_type"]
    output_dir = resolve_output_dir("prefix-tuning", context["experiment_name"])

    splits = load_splits(
        dataset=context["dataset"],
        revision=context["revision"],
        train_split=context["train_split"],
        eval_split=context["eval_split"],
        max_samples=context["max_samples"],
    )

    model, trainable = _load_prefix_model(model_name, task_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = _tokenize_split(splits[context["train_split"]], tokenizer, task_type)
    eval_ds = _tokenize_split(splits[context["eval_split"]], tokenizer, task_type) if context["eval_split"] else None

    trainer = _build_trainer(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=output_dir,
        epochs=context["epochs"],
        batch_size=context["batch_size"],
        learning_rate=context["learning_rate"],
        seed=context["seed"] or 42,
    )
    train_result = trainer.train()
    trainer.save_model(output_dir)

    metrics: dict[str, float] = {"train_loss": train_result.training_loss}
    if eval_ds:
        eval_metrics = trainer.evaluate()
        metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})

    for epoch in range(1, context["epochs"] + 1):
        log_epoch(epoch, context["epochs"], metrics["train_loss"])

    return TrainingAlgorithmResult(
        metrics=metrics,
        artifacts={"trainable_params": trainable, "num_virtual_tokens": _NUM_VIRTUAL_TOKENS, "checkpoint_dir": output_dir},
    )
