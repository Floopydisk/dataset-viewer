# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable, Mapping
from typing import Any, Optional, TypedDict


class TrainingExecutionContext(TypedDict):
    dataset: str
    revision: str
    model_name: str
    task_type: str
    train_split: str
    eval_split: Optional[str]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: Optional[int]
    max_samples: Optional[int]
    experiment_name: Optional[str]


class TrainingAlgorithmResult(TypedDict):
    metrics: Mapping[str, float]
    artifacts: Mapping[str, Any]


TrainingAlgorithm = Callable[[TrainingExecutionContext], TrainingAlgorithmResult]


def _not_implemented_algorithm(context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    raise NotImplementedError(
        "Training algorithm execution is not implemented yet. "
        f"Please implement algorithm for task '{context['task_type']}'."
    )


ALGORITHM_REGISTRY: dict[str, TrainingAlgorithm] = {
    "full-finetune": _not_implemented_algorithm,
    "lora": _not_implemented_algorithm,
    "qlora": _not_implemented_algorithm,
    "prefix-tuning": _not_implemented_algorithm,
    "prompt-tuning": _not_implemented_algorithm,
    "adapter-tuning": _not_implemented_algorithm,
    "linear-probing": _not_implemented_algorithm,
}


def run_training_algorithm(name: str, context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    algorithm = ALGORITHM_REGISTRY.get(name)
    if algorithm is None:
        supported = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unsupported training algorithm '{name}'. Supported algorithms: {supported}")
    return algorithm(context)
