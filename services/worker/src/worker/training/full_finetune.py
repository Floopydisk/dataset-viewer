# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""Full Fine-Tuning: update all model parameters end-to-end."""

import logging
from typing import Optional

from transformers import AutoTokenizer, PreTrainedModel

from worker.training._base import get_device, resolve_output_dir, resolve_task, set_seed
from worker.training._data import build_data_collator, ensure_padding_token, load_splits, tokenize_split
from worker.training._trainer import build_trainer, train_with_resume
from worker.training.algorithms import (
    TrainingAlgorithmResult,
    TrainingCancelledError,
    TrainingExecutionContext,
    build_cancellation_callback,
    build_progress_callback,
)


def _load_model(model_name: str, task_type: str) -> PreTrainedModel:
    model_class, _ = resolve_task(task_type)
    model: PreTrainedModel = model_class.from_pretrained(model_name)
    model.to(get_device())
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Full fine-tuning ({task_type}): {total:,} parameters trainable")
    return model


def run(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    set_seed(context["seed"])

    model_name = context["model_name"]
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    task_type = context["task_type"]
    output_dir = resolve_output_dir("full-finetune", context["experiment_name"], run_id=context["job_id"])

    splits = load_splits(
        dataset=context["dataset"],
        revision=context["revision"],
        train_split=context["train_split"],
        eval_split=context["eval_split"],
        max_samples=context["max_samples"],
        local_dataset_path=context["local_dataset_path"],
        local_dataset_format=context["local_dataset_format"],
    )

    model = _load_model(model_name, task_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_padding_token(tokenizer)
    data_collator = build_data_collator(tokenizer, task_type)

    train_ds = tokenize_split(splits[context["train_split"]], tokenizer, task_type)
    eval_ds = tokenize_split(splits[context["eval_split"]], tokenizer, task_type) if context["eval_split"] else None

    cancellation_callback = build_cancellation_callback(context)
    progress_callback = build_progress_callback(context)
    callbacks = [callback for callback in (cancellation_callback, progress_callback) if callback is not None]
    trainer = build_trainer(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=output_dir,
        data_collator=data_collator,
        epochs=context["epochs"],
        batch_size=context["batch_size"],
        learning_rate=context["learning_rate"],
        seed=context["seed"] or 42,
        callbacks=callbacks or None,
    )
    train_result, resume_metadata = train_with_resume(trainer=trainer, output_dir=output_dir)
    if cancellation_callback and cancellation_callback.cancelled:
        raise TrainingCancelledError(f"Training cancelled for job {context['job_id']}")
    trainer.save_model(output_dir)

    metrics: dict[str, float] = {"train_loss": train_result.training_loss}
    if eval_ds:
        eval_metrics = trainer.evaluate()
        metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})

    logging.info(f"Training complete after {context['epochs']} epochs: train_loss={metrics['train_loss']:.4f}")

    total = sum(p.numel() for p in model.parameters())
    return TrainingAlgorithmResult(
        metrics=metrics,
        artifacts={
            "total_params": total,
            "checkpoint_dir": output_dir,
            "modal_resumed_from_checkpoint": resume_metadata["resumed_from_checkpoint"],
            "modal_resume_checkpoint_path": resume_metadata["resume_checkpoint_path"],
        },
    )
