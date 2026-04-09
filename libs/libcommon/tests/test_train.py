# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.train import (
    SUPPORTED_TASK_TYPES,
    SUPPORTED_TRAINING_ALGORITHMS,
    TrainValidationError,
    get_training_capabilities,
    normalize_training_params,
    parse_training_request,
)


def test_normalize_training_params_accepts_aliases() -> None:
    params = normalize_training_params(
        {
            "experimentType": "baseline",
            "modelName": "bert-base-uncased",
            "epochs": "5",
            "batchSize": 16,
            "learningRate": "0.0005",
            "seed": "42",
            "taskType": "text-classification",
            "trainingAlgorithm": "lora",
            "trainSplit": "train",
            "evalSplit": "test",
            "maxSamples": "2000",
            "experimentName": "baseline-a",
        }
    )

    assert params == {
        "experiment_type": "baseline",
        "model_name": "bert-base-uncased",
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "seed": 42,
        "task_type": "text-classification",
        "training_algorithm": "lora",
        "train_split": "train",
        "eval_split": "test",
        "max_samples": 2000,
        "experiment_name": "baseline-a",
        "local_dataset_id": None,
        "local_dataset_path": None,
        "local_dataset_format": None,
    }


def test_normalize_training_params_rejects_unknown_keys() -> None:
    with pytest.raises(TrainValidationError, match="Unknown training parameter"):
        normalize_training_params({"optimizer": "adam"})


def test_parse_training_request_uses_defaults() -> None:
    parsed = parse_training_request(body={"dataset": "org/ds"})

    assert parsed["dataset"] == "org/ds"
    assert parsed["revision"] == "main"
    assert parsed["params_dict"] == {
        "experiment_type": None,
        "model_name": "tiny-bert",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "seed": None,
        "task_type": "text-classification",
        "training_algorithm": None,
        "train_split": "train",
        "eval_split": None,
        "max_samples": None,
        "experiment_name": None,
        "local_dataset_id": None,
        "local_dataset_path": None,
        "local_dataset_format": None,
    }


def test_parse_training_request_rejects_dataset_mismatch() -> None:
    with pytest.raises(TrainValidationError, match="must match"):
        parse_training_request(body={"dataset": "org/one"}, dataset_query="org/two")


def test_normalize_training_params_rejects_unsupported_task_type() -> None:
    with pytest.raises(TrainValidationError, match="Unsupported taskType"):
        normalize_training_params({"taskType": "image-classification"})


def test_normalize_training_params_rejects_unsupported_training_algorithm() -> None:
    with pytest.raises(TrainValidationError, match="Unsupported trainingAlgorithm"):
        normalize_training_params({"trainingAlgorithm": "my-custom-algo"})


def test_get_training_capabilities_returns_supported_values() -> None:
    capabilities = get_training_capabilities()

    assert capabilities["task_types"] == sorted(SUPPORTED_TASK_TYPES)
    assert capabilities["training_algorithms"] == sorted(SUPPORTED_TRAINING_ALGORITHMS)


def test_normalize_training_params_rejects_incompatible_algorithm_task_combo() -> None:
    with pytest.raises(TrainValidationError, match="Linear probing supports only"):
        normalize_training_params(
            {
                "trainingAlgorithm": "linear-probing",
                "taskType": "causal-lm",
                "modelName": "gpt2",
            }
        )


def test_normalize_training_params_rejects_incompatible_model_for_task() -> None:
    with pytest.raises(TrainValidationError, match="not compatible with taskType"):
        normalize_training_params(
            {
                "modelName": "gpt2",
                "taskType": "question-answering",
            }
        )


def test_parse_training_request_rejects_revision_with_spaces() -> None:
    with pytest.raises(TrainValidationError, match="cannot contain spaces"):
        parse_training_request(body={"dataset": "org/ds", "revision": "feature branch"})


def test_parse_training_request_rejects_local_eval_split_mismatch() -> None:
    with pytest.raises(TrainValidationError, match="must be empty or equal to 'trainSplit'"):
        parse_training_request(
            body={
                "datasetSource": "local",
                "localDatasetId": "dataset-1",
                "trainSplit": "train",
                "evalSplit": "validation",
            }
        )


def test_normalize_training_params_accepts_supported_experiment_type() -> None:
    params = normalize_training_params({"experimentType": "smoke-test"})
    assert params["experiment_type"] == "smoke-test"


def test_normalize_training_params_rejects_unsupported_experiment_type() -> None:
    with pytest.raises(TrainValidationError, match="Unsupported experimentType"):
        normalize_training_params({"experimentType": "chaos"})
