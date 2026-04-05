# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from typing import Any

import pytest

from worker.training.modal_backend import _build_compute_profile, _select_gpu_instance_class


def _context(
    *,
    model_name: str = "bert-base-uncased",
    task_type: str = "text-classification",
    batch_size: int = 16,
    epochs: int = 3,
) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "dataset": "org/dataset",
        "revision": "main",
        "model_name": model_name,
        "task_type": task_type,
        "train_split": "train",
        "eval_split": "validation",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "seed": 42,
        "max_samples": None,
        "experiment_name": "exp",
        "local_dataset_path": None,
        "local_dataset_format": None,
        "cancellation_checker": None,
        "progress_callback": None,
    }


@pytest.mark.parametrize(
    "algorithm,model_name,batch_size,task_type,expected_gpu",
    [
        ("linear-probing", "bert-base-uncased", 16, "text-classification", "T4"),
        ("prompt-tuning", "bert-base-uncased", 16, "text-classification", "T4"),
        ("prefix-tuning", "bert-base-uncased", 16, "text-classification", "T4"),
        ("adapter-tuning", "bert-base-uncased", 16, "text-classification", "T4"),
        ("lora", "bert-base-uncased", 16, "text-classification", "A10G"),
        ("full-finetune", "bert-base-uncased", 16, "text-classification", "A10G"),
        ("full-finetune", "meta-llama/Llama-3-8B", 16, "causal-lm", "A100"),
        ("full-finetune", "bert-base-uncased", 64, "text-classification", "A100"),
        ("qlora", "bert-base-uncased", 16, "text-classification", "A100"),
    ],
)
def test_select_gpu_instance_class(
    algorithm: str,
    model_name: str,
    batch_size: int,
    task_type: str,
    expected_gpu: str,
) -> None:
    context = _context(model_name=model_name, batch_size=batch_size, task_type=task_type)

    assert _select_gpu_instance_class(context=context, training_algorithm=algorithm) == expected_gpu


def test_build_compute_profile_uses_single_gpu() -> None:
    profile = _build_compute_profile(context=_context(), training_algorithm="lora")

    assert profile["provider"] == "nvidia"
    assert profile["gpu"] == "A10G"
    assert profile["gpu_count"] == 1
