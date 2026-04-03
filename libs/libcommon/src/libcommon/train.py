# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any, Optional, TypedDict


class TrainingParameters(TypedDict):
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: Optional[int]
    task_type: str
    training_algorithm: Optional[str]
    train_split: str
    eval_split: Optional[str]
    max_samples: Optional[int]
    experiment_name: Optional[str]


class TrainingRequest(TypedDict):
    dataset: str
    revision: str
    params_dict: TrainingParameters


class TrainValidationError(ValueError):
    pass


DEFAULT_MODEL_NAME = "tiny-bert"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3


_TRAIN_PARAM_ALIASES = {
    "modelName": "model_name",
    "model_name": "model_name",
    "epochs": "epochs",
    "batchSize": "batch_size",
    "batch_size": "batch_size",
    "learningRate": "learning_rate",
    "learning_rate": "learning_rate",
    "seed": "seed",
    "taskType": "task_type",
    "task_type": "task_type",
    "trainingAlgorithm": "training_algorithm",
    "training_algorithm": "training_algorithm",
    "trainSplit": "train_split",
    "train_split": "train_split",
    "evalSplit": "eval_split",
    "eval_split": "eval_split",
    "maxSamples": "max_samples",
    "max_samples": "max_samples",
    "experimentName": "experiment_name",
    "experiment_name": "experiment_name",
}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _to_bounded_int(value: Any, field_name: str, min_value: int, max_value: int) -> int:
    if isinstance(value, bool):
        raise TrainValidationError(f"'{field_name}' must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise TrainValidationError(f"'{field_name}' must be an integer")
        try:
            parsed = int(stripped)
        except ValueError as err:
            raise TrainValidationError(f"'{field_name}' must be an integer") from err
    else:
        raise TrainValidationError(f"'{field_name}' must be an integer")

    if parsed < min_value or parsed > max_value:
        raise TrainValidationError(f"'{field_name}' must be between {min_value} and {max_value}")
    return parsed


def _to_bounded_float(value: Any, field_name: str, min_value: float, max_value: float) -> float:
    if isinstance(value, bool):
        raise TrainValidationError(f"'{field_name}' must be a number")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise TrainValidationError(f"'{field_name}' must be a number")
        try:
            parsed = float(stripped)
        except ValueError as err:
            raise TrainValidationError(f"'{field_name}' must be a number") from err
    else:
        raise TrainValidationError(f"'{field_name}' must be a number")

    if parsed < min_value or parsed > max_value:
        raise TrainValidationError(f"'{field_name}' must be between {min_value} and {max_value}")
    return parsed


def normalize_training_params(params: Mapping[str, Any], strict: bool = True) -> TrainingParameters:
    normalized: dict[str, Any] = {}

    for key, value in params.items():
        canonical_key = _TRAIN_PARAM_ALIASES.get(key)
        if canonical_key is None:
            if strict:
                allowed = ", ".join(sorted(_TRAIN_PARAM_ALIASES))
                raise TrainValidationError(f"Unknown training parameter '{key}'. Allowed keys: {allowed}")
            continue
        normalized[canonical_key] = value

    model_name = normalized.get("model_name", DEFAULT_MODEL_NAME)
    if not _is_non_empty_string(model_name):
        raise TrainValidationError("'modelName' must be a non-empty string")

    epochs = _to_bounded_int(normalized.get("epochs", DEFAULT_EPOCHS), "epochs", min_value=1, max_value=100)
    batch_size = _to_bounded_int(
        normalized.get("batch_size", DEFAULT_BATCH_SIZE), "batchSize", min_value=1, max_value=2048
    )
    learning_rate = _to_bounded_float(
        normalized.get("learning_rate", DEFAULT_LEARNING_RATE), "learningRate", min_value=1e-7, max_value=10.0
    )

    seed_value = normalized.get("seed")
    seed: Optional[int]
    if seed_value is None:
        seed = None
    else:
        seed = _to_bounded_int(seed_value, "seed", min_value=0, max_value=2_147_483_647)

    task_type_value = normalized.get("task_type", "text-classification")
    if not _is_non_empty_string(task_type_value):
        raise TrainValidationError("'taskType' must be a non-empty string")

    training_algorithm_value = normalized.get("training_algorithm")
    training_algorithm: Optional[str]
    if training_algorithm_value is None:
        training_algorithm = None
    else:
        if not _is_non_empty_string(training_algorithm_value):
            raise TrainValidationError("'trainingAlgorithm' must be a non-empty string")
        training_algorithm = training_algorithm_value.strip()

    train_split_value = normalized.get("train_split", "train")
    if not _is_non_empty_string(train_split_value):
        raise TrainValidationError("'trainSplit' must be a non-empty string")

    eval_split_value = normalized.get("eval_split")
    eval_split: Optional[str]
    if eval_split_value is None:
        eval_split = None
    else:
        if not _is_non_empty_string(eval_split_value):
            raise TrainValidationError("'evalSplit' must be a non-empty string")
        eval_split = eval_split_value.strip()

    max_samples_value = normalized.get("max_samples")
    max_samples: Optional[int]
    if max_samples_value is None:
        max_samples = None
    else:
        max_samples = _to_bounded_int(max_samples_value, "maxSamples", min_value=1, max_value=100_000_000)

    experiment_name_value = normalized.get("experiment_name")
    experiment_name: Optional[str]
    if experiment_name_value is None:
        experiment_name = None
    else:
        if not _is_non_empty_string(experiment_name_value):
            raise TrainValidationError("'experimentName' must be a non-empty string")
        experiment_name = experiment_name_value.strip()

    return {
        "model_name": model_name.strip(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "task_type": task_type_value.strip(),
        "training_algorithm": training_algorithm,
        "train_split": train_split_value.strip(),
        "eval_split": eval_split,
        "max_samples": max_samples,
        "experiment_name": experiment_name,
    }


def parse_training_request(body: Mapping[str, Any], dataset_query: Optional[str] = None) -> TrainingRequest:
    if not isinstance(body, Mapping):
        raise TrainValidationError("Request body must be a JSON object")

    body_dataset = body.get("dataset")
    if dataset_query and body_dataset and dataset_query != body_dataset:
        raise TrainValidationError("'dataset' in query and body must match")

    dataset = dataset_query or body_dataset
    if not _is_non_empty_string(dataset):
        raise TrainValidationError("'dataset' is required")

    revision_value = body.get("revision", "main")
    if not _is_non_empty_string(revision_value):
        raise TrainValidationError("'revision' must be a non-empty string")

    training_payload = {
        key: value
        for key, value in body.items()
        if key not in {"dataset", "revision"}
    }
    params_dict = normalize_training_params(training_payload, strict=True)

    return {
        "dataset": dataset.strip(),
        "revision": revision_value.strip(),
        "params_dict": params_dict,
    }