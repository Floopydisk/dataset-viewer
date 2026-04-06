# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from typing import Any, Optional

import torch
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
    bf16: Optional[bool] = None,
    tf32: Optional[bool] = None,
) -> Trainer:
    use_fp16 = False
    use_bf16 = False
    use_tf32 = False

    if get_device().type == "cuda":
        default_bf16 = torch.cuda.is_bf16_supported()
        use_bf16 = default_bf16 if bf16 is None else bf16
        use_fp16 = (not use_bf16) if fp16 is None else fp16

        # Ampere+ (SM80+) benefits from TF32 tensor cores for matmul-heavy workloads.
        capability = torch.cuda.get_device_capability()
        default_tf32 = capability[0] >= 8
        use_tf32 = default_tf32 if tf32 is None else tf32

    if use_bf16 and use_fp16:
        # TrainingArguments cannot simultaneously enable both mixed precision modes.
        use_fp16 = False

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
        bf16=use_bf16,
        tf32=use_tf32,
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