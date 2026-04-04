# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""QLoRA: Parameter-efficient LoRA training (CPU-only for development phase)."""

import logging
from typing import Any, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments

from worker.training._base import resolve_output_dir, resolve_task, set_seed
from worker.training._data import build_data_collator, ensure_padding_token, load_splits, tokenize_split
from worker.training.algorithms import TrainingAlgorithmResult, TrainingExecutionContext

_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05


def _load_qlora_model(model_name: str, task_type: str) -> tuple[Any, int]:
    """Load model with LoRA (CPU-only for development phase).

    Returns:
        (model, trainable_param_count)
    """
    model_class, peft_task_type = resolve_task(task_type)
    lora_config = LoraConfig(
        task_type=peft_task_type,
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
    )

    base = model_class.from_pretrained(model_name)
    base.to(torch.device("cpu"))
    logging.info(f"QLoRA ({task_type}): LoRA (CPU-only) enabled")

    model = get_peft_model(base, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logging.info(f"QLoRA ({task_type}): {trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.2f}%)")
    return model, trainable


def _build_trainer(
    model: Any,
    train_ds: Dataset,
    eval_ds: Optional[Dataset],
    output_dir: str,
    data_collator: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    use_fp16: bool,
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
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=data_collator)


def run(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    set_seed(context["seed"])

    model_name = context["model_name"]
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    task_type = context["task_type"]
    output_dir = resolve_output_dir("qlora", context["experiment_name"])

    splits = load_splits(
        dataset=context["dataset"],
        revision=context["revision"],
        train_split=context["train_split"],
        eval_split=context["eval_split"],
        max_samples=context["max_samples"],
    )

    model, trainable = _load_qlora_model(model_name, task_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_padding_token(tokenizer)
    data_collator = build_data_collator(tokenizer, task_type)

    train_ds = tokenize_split(splits[context["train_split"]], tokenizer, task_type)
    eval_ds = tokenize_split(splits[context["eval_split"]], tokenizer, task_type) if context["eval_split"] else None

    trainer = _build_trainer(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        output_dir=output_dir,
        data_collator=data_collator,
        epochs=context["epochs"],
        batch_size=context["batch_size"],
        learning_rate=context["learning_rate"],
        seed=context["seed"] or 42,
        use_fp16=False,
    )
    train_result = trainer.train()
    trainer.save_model(output_dir)

    metrics: dict[str, float] = {"train_loss": train_result.training_loss}
    if eval_ds:
        eval_metrics = trainer.evaluate()
        metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})

    logging.info(f"Training complete after {context['epochs']} epochs: train_loss={metrics['train_loss']:.4f}")

    return TrainingAlgorithmResult(
        metrics=metrics,
        artifacts={
            "trainable_params": trainable,
            "quantization": "nf4" if quantized else "none",
            "lora_r": _LORA_R,
            "checkpoint_dir": output_dir,
        },
    )
