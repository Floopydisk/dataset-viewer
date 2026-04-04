# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Callable, Mapping
import importlib
import inspect
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
    local_dataset_path: Optional[str]
    local_dataset_format: Optional[str]


class TrainingAlgorithmResult(TypedDict):
    metrics: Mapping[str, float]
    artifacts: Mapping[str, Any]


TrainingAlgorithm = Callable[[TrainingExecutionContext], TrainingAlgorithmResult]


def _build_registry() -> dict[str, TrainingAlgorithm]:
    # Deferred imports so algorithm modules can import from this module without circular deps
    from worker.training import (
        adapter_tuning,
        full_finetune,
        linear_probing,
        lora,
        prefix_tuning,
        prompt_tuning,
        qlora,
    )

    return {
        "linear-probing": linear_probing.run,
        "full-finetune": full_finetune.run,
        "lora": lora.run,
        "qlora": qlora.run,
        "prefix-tuning": prefix_tuning.run,
        "prompt-tuning": prompt_tuning.run,
        "adapter-tuning": adapter_tuning.run,
    }


def run_training_algorithm(name: str, context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    # Transformers Trainer expects accelerate.Accelerator.unwrap_model to support
    # keep_torch_compile; older accelerate versions raise a runtime TypeError.
    accelerator_module = importlib.import_module("accelerate")
    accelerator_class = getattr(accelerator_module, "Accelerator")
    unwrap_signature = inspect.signature(accelerator_class.unwrap_model)
    if "keep_torch_compile" not in unwrap_signature.parameters:
        raise RuntimeError(
            "Incompatible accelerate runtime detected. Upgrade accelerate to >=0.31.0 to run training jobs."
        )

    registry = _build_registry()
    algorithm = registry.get(name)
    if algorithm is None:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported training algorithm '{name}'. Supported algorithms: {supported}")
    return algorithm(context)


# Expose supported names without importing algorithm modules
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    ["linear-probing", "full-finetune", "lora", "qlora", "prefix-tuning", "prompt-tuning", "adapter-tuning"]
)
