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

    return {
        "model_name": model_name.strip(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
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