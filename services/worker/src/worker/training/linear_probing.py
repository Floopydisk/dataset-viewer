# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""Linear Probing: freeze all backbone parameters, train only the classification head."""

import logging
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments

from worker.training._base import get_device, log_epoch, resolve_output_dir, resolve_task, set_seed
from worker.training._data import load_splits, resolve_text_column, resolve_token_column
from worker.training.algorithms import TrainingAlgorithmResult, TrainingExecutionContext

_SUPPORTED_TASK_TYPES = frozenset(["text-classification", "token-classification"])


def _load_frozen_model(model_name: str, task_type: str) -> tuple[PreTrainedModel, int]:
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Linear probing does not support task_type '{task_type}'. "
            f"Supported: {sorted(_SUPPORTED_TASK_TYPES)}"
        )
    model_class, _ = resolve_task(task_type)
    model: PreTrainedModel = model_class.from_pretrained(model_name)
    model.to(get_device())
    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Linear probing: {trainable:,} / {total:,} parameters trainable")
    return model, trainable


def _tokenize_split(ds: Dataset, tokenizer: Any, task_type: str) -> Dataset:
    if task_type == "token-classification":
        text_column = resolve_token_column(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(  # type: ignore[operator]
                batch[text_column],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
            )
    else:
        text_column = resolve_text_column(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(  # type: ignore[operator]
                batch[text_column], truncation=True, padding="max_length", max_length=128
            )

    return ds.map(_tok, batched=True)


def _build_trainer(
    model: PreTrainedModel,
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


def _collect_metrics(trainer: Trainer, eval_ds: Optional[Dataset], train_loss: float) -> dict[str, float]:
    metrics: dict[str, float] = {"train_loss": train_loss}
    if eval_ds:
        eval_metrics = trainer.evaluate()
        metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})
    return metrics


def run(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    set_seed(context["seed"])

    model_name = context["model_name"]
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    task_type = context["task_type"]
    output_dir = resolve_output_dir("linear-probing", context["experiment_name"])

    splits = load_splits(
        dataset=context["dataset"],
        revision=context["revision"],
        train_split=context["train_split"],
        eval_split=context["eval_split"],
        max_samples=context["max_samples"],
    )

    model, trainable = _load_frozen_model(model_name, task_type)
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

    metrics = _collect_metrics(trainer, eval_ds, train_result.training_loss)

    for epoch in range(1, context["epochs"] + 1):
        log_epoch(epoch, context["epochs"], metrics["train_loss"])

    return TrainingAlgorithmResult(
        metrics=metrics,
        artifacts={"trainable_params": trainable, "checkpoint_dir": output_dir},
    )
