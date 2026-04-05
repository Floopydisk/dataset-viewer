# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from typing import Any, Optional

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from worker.training._base import get_device


def build_trainer(
    *,
    model: Any,
    train_ds: Dataset,
    eval_ds: Optional[Dataset],
    output_dir: str,
    data_collator: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    callbacks: Optional[list[Any]],
    fp16: Optional[bool] = None,
) -> Trainer:
    use_fp16 = get_device().type == "cuda" if fp16 is None else fp16
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
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )