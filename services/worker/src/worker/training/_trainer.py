# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
from pathlib import Path
import re
from typing import Any, Optional, TypedDict

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from worker.training._base import get_device


_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


class ResumeMetadata(TypedDict):
    resumed_from_checkpoint: bool
    resume_checkpoint_path: Optional[str]


def _latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None

    latest_step = -1
    latest_dir: Optional[Path] = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_DIR_RE.match(child.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_dir = child

    return str(latest_dir) if latest_dir is not None else None


def train_with_resume(*, trainer: Trainer, output_dir: str) -> tuple[Any, ResumeMetadata]:
    checkpoint_dir = _latest_checkpoint_dir(output_dir)
    if checkpoint_dir is None:
        return trainer.train(), {
            "resumed_from_checkpoint": False,
            "resume_checkpoint_path": None,
        }

    logging.info("Resuming training from checkpoint: %s", checkpoint_dir)
    try:
        return trainer.train(resume_from_checkpoint=checkpoint_dir), {
            "resumed_from_checkpoint": True,
            "resume_checkpoint_path": checkpoint_dir,
        }
    except Exception as err:
        logging.warning("Checkpoint resume failed from %s (%s). Restarting from scratch.", checkpoint_dir, err)
        return trainer.train(), {
            "resumed_from_checkpoint": False,
            "resume_checkpoint_path": checkpoint_dir,
        }


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