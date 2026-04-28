# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""QLoRA: Parameter-efficient LoRA training with a GPU quantized path and CPU fallback."""

import logging
from typing import Any, Optional

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig

from worker.training._base import resolve_output_dir, resolve_task, set_seed
from worker.training._data import (
    build_data_collator,
    ensure_padding_token,
    load_splits,
    resolve_eval_split_name,
    tokenize_split,
)
from worker.training._trainer import build_trainer, train_with_resume
from worker.training.algorithms import (
    TrainingAlgorithmResult,
    TrainingCancelledError,
    TrainingExecutionContext,
    build_cancellation_callback,
    build_progress_callback,
)

_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05


def _load_qlora_model(model_name: str, task_type: str) -> tuple[Any, int, bool]:
    """Load a quantized QLoRA model on GPU when available, otherwise fall back to CPU.

    Returns:
        (model, trainable_param_count, gpu_path_enabled)
    """
    model_class, peft_task_type = resolve_task(task_type)
    lora_config = LoraConfig(
        task_type=peft_task_type,
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
    )

    import torch

    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = model_class.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
        base = prepare_model_for_kbit_training(base)
        gpu_path_enabled = True
        logging.info(f"QLoRA ({task_type}): GPU path enabled with 4-bit NF4 quantization")
    else:
        base = model_class.from_pretrained(model_name)
        base.to(torch.device("cpu"))
        gpu_path_enabled = False
        logging.info(f"QLoRA ({task_type}): LoRA (CPU-only) enabled")

    model = get_peft_model(base, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logging.info(f"QLoRA ({task_type}): {trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.2f}%)")
    return model, trainable, gpu_path_enabled


def run(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    set_seed(context["seed"])

    model_name = context["model_name"]
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    task_type = context["task_type"]
    output_dir = resolve_output_dir("qlora", context["experiment_name"], run_id=context["job_id"])

    eval_split_name = resolve_eval_split_name(context["eval_split"], context.get("test_split_ratio"))
    splits = load_splits(
        dataset=context["dataset"],
        revision=context["revision"],
        train_split=context["train_split"],
        eval_split=context["eval_split"],
        test_split_ratio=context.get("test_split_ratio"),
        max_samples=context["max_samples"],
        seed=context.get("seed"),
        local_dataset_path=context["local_dataset_path"],
        local_dataset_format=context["local_dataset_format"],
    )

    model, trainable, quantized = _load_qlora_model(model_name, task_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_padding_token(tokenizer)
    data_collator = build_data_collator(tokenizer, task_type)

    train_ds = tokenize_split(splits[context["train_split"]], tokenizer, task_type)
    eval_ds = tokenize_split(splits[eval_split_name], tokenizer, task_type) if eval_split_name else None

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

    return TrainingAlgorithmResult(
        metrics=metrics,
        artifacts={
            "trainable_params": trainable,
            "quantization": "4bit-nf4" if quantized else "none",
            "lora_r": _LORA_R,
            "checkpoint_dir": output_dir,
            "modal_resumed_from_checkpoint": resume_metadata["resumed_from_checkpoint"],
            "modal_resume_checkpoint_path": resume_metadata["resume_checkpoint_path"],
        },
    )
