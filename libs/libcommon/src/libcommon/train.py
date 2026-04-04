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
    local_dataset_id: Optional[str]
    local_dataset_path: Optional[str]
    local_dataset_format: Optional[str]


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

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(
    {
        "text-classification",
        "token-classification",
        "seq2seq",
        "summarization",
        "causal-lm",
        "question-answering",
    }
)

SUPPORTED_TRAINING_ALGORITHMS: frozenset[str] = frozenset(
    {
        "linear-probing",
        "full-finetune",
        "lora",
        "qlora",
        "prefix-tuning",
        "prompt-tuning",
        "adapter-tuning",
    }
)


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
    "localDatasetId": "local_dataset_id",
    "local_dataset_id": "local_dataset_id",
    "localDatasetPath": "local_dataset_path",
    "local_dataset_path": "local_dataset_path",
    "localDatasetFormat": "local_dataset_format",
    "local_dataset_format": "local_dataset_format",
    "datasetSource": "dataset_source",
    "dataset_source": "dataset_source",
}

SUPPORTED_TRAINING_DATASET_SOURCES: frozenset[str] = frozenset({"huggingface", "local"})
SUPPORTED_LOCAL_DATASET_FORMATS: frozenset[str] = frozenset({"csv", "json", "jsonl", "parquet"})


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


def get_training_capabilities() -> dict[str, list[str]]:
    return {
        "task_types": sorted(SUPPORTED_TASK_TYPES),
        "training_algorithms": sorted(SUPPORTED_TRAINING_ALGORITHMS),
    }


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
    task_type = task_type_value.strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        raise TrainValidationError(f"Unsupported taskType '{task_type}'. Supported: {supported}")

    training_algorithm_value = normalized.get("training_algorithm")
    training_algorithm: Optional[str]
    if training_algorithm_value is None:
        training_algorithm = None
    else:
        if not _is_non_empty_string(training_algorithm_value):
            raise TrainValidationError("'trainingAlgorithm' must be a non-empty string")
        training_algorithm = training_algorithm_value.strip()
        if training_algorithm not in SUPPORTED_TRAINING_ALGORITHMS:
            supported = ", ".join(sorted(SUPPORTED_TRAINING_ALGORITHMS))
            raise TrainValidationError(
                f"Unsupported trainingAlgorithm '{training_algorithm}'. Supported: {supported}"
            )

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

    local_dataset_id_value = normalized.get("local_dataset_id")
    local_dataset_id: Optional[str]
    if local_dataset_id_value is None:
        local_dataset_id = None
    else:
        if not _is_non_empty_string(local_dataset_id_value):
            raise TrainValidationError("'localDatasetId' must be a non-empty string")
        local_dataset_id = local_dataset_id_value.strip()

    local_dataset_path_value = normalized.get("local_dataset_path")
    local_dataset_path: Optional[str]
    if local_dataset_path_value is None:
        local_dataset_path = None
    else:
        if not _is_non_empty_string(local_dataset_path_value):
            raise TrainValidationError("'localDatasetPath' must be a non-empty string")
        local_dataset_path = local_dataset_path_value.strip()

    local_dataset_format_value = normalized.get("local_dataset_format")
    local_dataset_format: Optional[str]
    if local_dataset_format_value is None:
        local_dataset_format = None
    else:
        if not _is_non_empty_string(local_dataset_format_value):
            raise TrainValidationError("'localDatasetFormat' must be a non-empty string")
        local_dataset_format = local_dataset_format_value.strip().lower()
        if local_dataset_format not in SUPPORTED_LOCAL_DATASET_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_LOCAL_DATASET_FORMATS))
            raise TrainValidationError(
                f"Unsupported localDatasetFormat '{local_dataset_format}'. Supported: {supported}"
            )

    return {
        "model_name": model_name.strip(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "task_type": task_type,
        "training_algorithm": training_algorithm,
        "train_split": train_split_value.strip(),
        "eval_split": eval_split,
        "max_samples": max_samples,
        "experiment_name": experiment_name,
        "local_dataset_id": local_dataset_id,
        "local_dataset_path": local_dataset_path,
        "local_dataset_format": local_dataset_format,
    }


def parse_training_request(body: Mapping[str, Any], dataset_query: Optional[str] = None) -> TrainingRequest:
    if not isinstance(body, Mapping):
        raise TrainValidationError("Request body must be a JSON object")

    dataset_source_value = body.get("datasetSource", body.get("dataset_source", "huggingface"))
    if not _is_non_empty_string(dataset_source_value):
        raise TrainValidationError("'datasetSource' must be a non-empty string")
    dataset_source = dataset_source_value.strip().lower()
    if dataset_source not in SUPPORTED_TRAINING_DATASET_SOURCES:
        supported_sources = ", ".join(sorted(SUPPORTED_TRAINING_DATASET_SOURCES))
        raise TrainValidationError(f"Unsupported datasetSource '{dataset_source}'. Supported: {supported_sources}")

    body_dataset = body.get("dataset")
    if dataset_query and body_dataset and dataset_query != body_dataset:
        raise TrainValidationError("'dataset' in query and body must match")

    dataset = dataset_query or body_dataset

    revision_value = body.get("revision", "main")
    if not _is_non_empty_string(revision_value):
        raise TrainValidationError("'revision' must be a non-empty string")

    training_payload = {
        key: value
        for key, value in body.items()
        if key not in {"dataset", "revision"}
    }
    params_dict = normalize_training_params(training_payload, strict=True)

    if dataset_source == "local":
        local_dataset_id = params_dict.get("local_dataset_id")
        if not local_dataset_id:
            if _is_non_empty_string(dataset):
                local_dataset_id = dataset.strip()
            else:
                raise TrainValidationError("'localDatasetId' is required when datasetSource is 'local'")
            params_dict["local_dataset_id"] = local_dataset_id
        dataset = f"local://pending/{local_dataset_id}"
    else:
        if not _is_non_empty_string(dataset):
            raise TrainValidationError("'dataset' is required")

    return {
        "dataset": dataset.strip(),
        "revision": revision_value.strip(),
        "params_dict": params_dict,
    }