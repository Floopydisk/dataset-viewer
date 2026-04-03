# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

"""QLoRA: LoRA on a 4-bit NF4-quantized base model (falls back to standard LoRA on CPU)."""

import logging
from typing import Any, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, Trainer, TrainingArguments

from worker.training._base import log_epoch, resolve_output_dir, resolve_task, set_seed
from worker.training._data import (
    load_splits,
    resolve_qa_columns,
    resolve_seq2seq_columns,
    resolve_text_column,
    resolve_token_column,
)
from worker.training.algorithms import TrainingAlgorithmResult, TrainingExecutionContext

_LORA_R = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.05


def _load_qlora_model(model_name: str, task_type: str) -> tuple[Any, int, bool]:
    """Load model with 4-bit quantization when CUDA is available, plain LoRA otherwise.

    Returns:
        (model, trainable_param_count, quantized)
    """
    model_class, peft_task_type = resolve_task(task_type)
    lora_config = LoraConfig(
        task_type=peft_task_type,
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
    )

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base: PreTrainedModel = model_class.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        base = prepare_model_for_kbit_training(base)
        quantized = True
        logging.info(f"QLoRA ({task_type}): 4-bit NF4 quantization enabled")
    else:
        logging.warning(
            "QLoRA: CUDA not available — falling back to standard LoRA in full precision. "
            "4-bit quantization requires a CUDA-capable GPU."
        )
        base = model_class.from_pretrained(model_name)
        base.to(torch.device("cpu"))
        quantized = False

    model = get_peft_model(base, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logging.info(f"QLoRA ({task_type}): {trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.2f}%)")
    return model, trainable, quantized


def _tokenize_split(ds: Dataset, tokenizer: Any, task_type: str) -> Dataset:
    if task_type == "token-classification":
        col = resolve_token_column(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(  # type: ignore[operator]
                batch[col], truncation=True, padding="max_length", max_length=128, is_split_into_words=True
            )

    elif task_type == "seq2seq":
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

    elif task_type == "question-answering":
        ctx_col, q_col = resolve_qa_columns(ds)

        def _tok(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(  # type: ignore[operator]
                batch[q_col], batch[ctx_col], truncation=True, padding="max_length", max_length=384
            )

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
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)


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

    model, trainable, quantized = _load_qlora_model(model_name, task_type)
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
        use_fp16=quantized,
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
        artifacts={
            "trainable_params": trainable,
            "quantization": "nf4" if quantized else "none",
            "lora_r": _LORA_R,
            "checkpoint_dir": output_dir,
        },
    )
