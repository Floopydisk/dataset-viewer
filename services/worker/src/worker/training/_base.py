# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import os
import random
from typing import Optional, Union

import torch
from peft import TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

# Maps task_type string → (AutoModel class, PEFT TaskType)
_TASK_MAP: dict[
    str,
    tuple[
        Union[
            type[AutoModelForSequenceClassification],
            type[AutoModelForTokenClassification],
            type[AutoModelForSeq2SeqLM],
            type[AutoModelForCausalLM],
            type[AutoModelForQuestionAnswering],
        ],
        TaskType,
    ],
] = {
    "text-classification": (AutoModelForSequenceClassification, TaskType.SEQ_CLS),
    "token-classification": (AutoModelForTokenClassification, TaskType.TOKEN_CLS),
    "seq2seq": (AutoModelForSeq2SeqLM, TaskType.SEQ_2_SEQ_LM),
    "summarization": (AutoModelForSeq2SeqLM, TaskType.SEQ_2_SEQ_LM),
    "causal-lm": (AutoModelForCausalLM, TaskType.CAUSAL_LM),
    "question-answering": (AutoModelForQuestionAnswering, TaskType.QUESTION_ANS),
}

_SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(_TASK_MAP)


def resolve_task(task_type: str) -> tuple[type[PreTrainedModel], TaskType]:
    """Return (AutoModel class, PEFT TaskType) for a given task_type string.

    Raises:
        ValueError: if task_type is not in the supported set.
    """
    entry = _TASK_MAP.get(task_type)
    if entry is None:
        supported = ", ".join(sorted(_SUPPORTED_TASK_TYPES))
        raise ValueError(f"Unsupported task_type '{task_type}'. Supported: {supported}")
    return entry  # type: ignore[return-value]


def resolve_output_dir(algorithm: str, experiment_name: Optional[str]) -> str:
    """Return a deterministic output directory path for saving checkpoints."""
    if experiment_name:
        path = os.path.join("/tmp", "training", algorithm, experiment_name.strip())
    else:
        path = os.path.join("/tmp", "training", algorithm)
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
