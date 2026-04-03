# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from worker.training.algorithms import run_training_algorithm


def _get_context() -> dict[str, object]:
    return {
        "dataset": "stanfordnlp/imdb",
        "revision": "main",
        "model_name": "distilbert-base-uncased",
        "task_type": "text-classification",
        "train_split": "train",
        "eval_split": "test",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "seed": 42,
        "max_samples": 100,
        "experiment_name": "smoke",
    }


def test_run_training_algorithm_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unsupported training algorithm"):
        run_training_algorithm(name="my-algo", context=_get_context())


def test_run_training_algorithm_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        run_training_algorithm(name="lora", context=_get_context())
