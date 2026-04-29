# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any, Optional, TypedDict


class TrainingParameters(TypedDict):
    experiment_type: Optional[str]
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: Optional[int]
    task_type: str
    training_algorithm: Optional[str]
    train_split: str
    eval_split: Optional[str]
    test_split_ratio: Optional[float]
    max_samples: Optional[int]
    experiment_name: Optional[str]
    local_dataset_id: Optional[str]
    local_dataset_path: Optional[str]
    local_dataset_format: Optional[str]
    # Resource allocation
    use_gpu: Optional[bool]
    gpu_count: Optional[int]
    gpu_type: Optional[str]
    cpu_cores: Optional[int]
    memory_gb: Optional[int]


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
DEFAULT_TEST_SPLIT_RATIO = 0.3
MIN_TEST_SPLIT_RATIO = 0.05
MAX_TEST_SPLIT_RATIO = 0.5
DEFAULT_MAX_SAMPLES = None
DEFAULT_SEED = None

RECOMMENDED_MODELS_BY_TASK: dict[str, list[str]] = {
    "text-classification": ["bert-base-uncased", "distilbert-base-uncased"],
    "token-classification": ["bert-base-cased", "distilbert-base-cased"],
    "seq2seq": ["google-t5/t5-small", "facebook/bart-base"],
    "summarization": ["google-t5/t5-small", "facebook/bart-base"],
    "causal-lm": ["gpt2", "distilgpt2"],
    "question-answering": [
        "distilbert-base-cased-distilled-squad",
        "bert-large-uncased-whole-word-masking-finetuned-squad",
    ],
}

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

SUPPORTED_EXPERIMENT_TYPES: frozenset[str] = frozenset(
    {
        "baseline",
        "ablation",
        "comparison",
        "smoke-test",
    }
)

_LINEAR_PROBING_SUPPORTED_TASKS: frozenset[str] = frozenset(
    {"text-classification", "token-classification"}
)
_PREFIX_PROMPT_SUPPORTED_TASKS: frozenset[str] = frozenset(
    {"seq2seq", "summarization", "causal-lm", "text-classification"}
)


_TRAIN_PARAM_ALIASES = {
    "experimentType": "experiment_type",
    "experiment_type": "experiment_type",
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
    "testSplitRatio": "test_split_ratio",
    "test_split_ratio": "test_split_ratio",
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

# Resource-related aliases (frontend uses camelCase)
_TRAIN_PARAM_ALIASES.update({
    "useGPU": "use_gpu",
    "gpuCount": "gpu_count",
    "gpuType": "gpu_type",
    "cpuCores": "cpu_cores",
    "memoryGb": "memory_gb",
})

SUPPORTED_TRAINING_DATASET_SOURCES: frozenset[str] = frozenset({"huggingface", "local"})
SUPPORTED_LOCAL_DATASET_FORMATS: frozenset[str] = frozenset({"csv", "json", "jsonl", "parquet"})


def _validate_no_whitespace(value: str, field_name: str) -> str:
    if any(char.isspace() for char in value):
        raise TrainValidationError(f"'{field_name}' cannot contain spaces")
    return value


def _validate_revision(value: str) -> str:
    revision = value.strip()
    if not revision:
        raise TrainValidationError("'revision' must be a non-empty string")
    if len(revision) > 128:
        raise TrainValidationError("'revision' must be at most 128 characters")
    _validate_no_whitespace(revision, "revision")
    if revision.startswith("/") or revision.endswith("/"):
        raise TrainValidationError("'revision' cannot start or end with '/'")
    return revision


def _validate_split_name(value: str, field_name: str) -> str:
    split = value.strip()
    if not split:
        raise TrainValidationError(f"'{field_name}' must be a non-empty string")
    if len(split) > 128:
        raise TrainValidationError(f"'{field_name}' must be at most 128 characters")
    _validate_no_whitespace(split, field_name)
    return split


def _is_model_task_compatible(model_name: str, task_type: str) -> bool:
    normalized_model = model_name.strip().lower()
    if not normalized_model:
        return False

    causal_markers = (
        "gpt",
        "llama",
        "mistral",
        "mixtral",
        "qwen",
        "falcon",
        "gemma",
        "phi",
        "opt",
        "bloom",
        "rwkv",
    )
    seq2seq_markers = (
        "t5",
        "bart",
        "pegasus",
        "mbart",
        "flan",
    )

    is_causal_model = any(marker in normalized_model for marker in causal_markers)
    is_seq2seq_model = any(marker in normalized_model for marker in seq2seq_markers)

    if task_type == "causal-lm":
        return is_causal_model
    if task_type in {"seq2seq", "summarization"}:
        return is_seq2seq_model
    if task_type == "question-answering":
        return not is_causal_model
    return True


def _validate_algorithm_task_compatibility(task_type: str, training_algorithm: Optional[str]) -> None:
    if training_algorithm == "linear-probing" and task_type not in _LINEAR_PROBING_SUPPORTED_TASKS:
        raise TrainValidationError(
            "Linear probing supports only text-classification and token-classification tasks. "
            f"Received taskType '{task_type}'."
        )
    if training_algorithm in {"prefix-tuning", "prompt-tuning"} and task_type not in _PREFIX_PROMPT_SUPPORTED_TASKS:
        raise TrainValidationError(
            f"{training_algorithm} does not support taskType '{task_type}'. "
            f"Supported: {sorted(_PREFIX_PROMPT_SUPPORTED_TASKS)}"
        )


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


def _validate_resource_allocation(normalized: dict[str, Any]) -> None:
    # use_gpu can be None or truthy; coerce booleans handled at caller if needed
    use_gpu = normalized.get("use_gpu")
    if use_gpu:
        gpu_count = normalized.get("gpu_count")
        if gpu_count is None:
            raise TrainValidationError("'gpuCount' is required when 'useGPU' is true")
        normalized["gpu_count"] = _to_bounded_int(gpu_count, "gpuCount", 1, 64)
        gpu_type = normalized.get("gpu_type")
        if gpu_type is not None:
            if not _is_non_empty_string(gpu_type):
                raise TrainValidationError("'gpuType' must be a non-empty string")
            if len(gpu_type.strip()) > 100:
                raise TrainValidationError("'gpuType' must be at most 100 characters")

    cpu_cores = normalized.get("cpu_cores")
    if cpu_cores is not None:
        normalized["cpu_cores"] = _to_bounded_int(cpu_cores, "cpuCores", 1, 256)

    memory_gb = normalized.get("memory_gb")
    if memory_gb is not None:
        normalized["memory_gb"] = _to_bounded_int(memory_gb, "memoryGb", 1, 4096)


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


def get_training_capabilities() -> dict[str, Any]:
    return {
        "task_types": sorted(SUPPORTED_TASK_TYPES),
        "training_algorithms": sorted(SUPPORTED_TRAINING_ALGORITHMS),
        "experiment_types": sorted(SUPPORTED_EXPERIMENT_TYPES),
        "dataset_sources": sorted(SUPPORTED_TRAINING_DATASET_SOURCES),
        "local_dataset_formats": sorted(SUPPORTED_LOCAL_DATASET_FORMATS),
        "models": sorted(
            {DEFAULT_MODEL_NAME}
            | {model for models in RECOMMENDED_MODELS_BY_TASK.values() for model in models}
        ),
        "default_model": DEFAULT_MODEL_NAME,
        "recommended_models": RECOMMENDED_MODELS_BY_TASK,
        "defaults": {
            "model_name": DEFAULT_MODEL_NAME,
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "task_type": "text-classification",
            "train_split": "train",
            "eval_split": None,
            "test_split_ratio": DEFAULT_TEST_SPLIT_RATIO,
            "max_samples": DEFAULT_MAX_SAMPLES,
            "seed": DEFAULT_SEED,
        },
        "ranges": {
            "epochs": {"min": 1, "max": 100},
            "batch_size": {"min": 1, "max": 2048},
            "learning_rate": {"min": 1e-7, "max": 10.0},
            "test_split_ratio": {"min": MIN_TEST_SPLIT_RATIO, "max": MAX_TEST_SPLIT_RATIO},
            "max_samples": {"min": 1, "max": 100_000_000},
            "seed": {"min": 0, "max": 2_147_483_647},
            "gpu_count": {"min": 1, "max": 64},
            "cpu_cores": {"min": 1, "max": 256},
            "memory_gb": {"min": 1, "max": 4096},
        },
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

    experiment_type_value = normalized.get("experiment_type")
    experiment_type: Optional[str]
    if experiment_type_value is None:
        experiment_type = None
    else:
        if not _is_non_empty_string(experiment_type_value):
            raise TrainValidationError("'experimentType' must be a non-empty string")
        experiment_type = experiment_type_value.strip().lower()
        if experiment_type not in SUPPORTED_EXPERIMENT_TYPES:
            supported = ", ".join(sorted(SUPPORTED_EXPERIMENT_TYPES))
            raise TrainValidationError(
                f"Unsupported experimentType '{experiment_type}'. Supported: {supported}"
            )

    model_name = normalized.get("model_name", DEFAULT_MODEL_NAME)
    if not _is_non_empty_string(model_name):
        raise TrainValidationError("'modelName' must be a non-empty string")
    if len(model_name.strip()) > 200:
        raise TrainValidationError("'modelName' must be at most 200 characters")

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
    train_split = _validate_split_name(train_split_value, "trainSplit")

    eval_split_value = normalized.get("eval_split")
    eval_split: Optional[str]
    if eval_split_value is None:
        eval_split = None
    else:
        if not _is_non_empty_string(eval_split_value):
            raise TrainValidationError("'evalSplit' must be a non-empty string")
        eval_split = _validate_split_name(eval_split_value, "evalSplit")

    test_split_ratio_value = normalized.get("test_split_ratio")
    test_split_ratio: Optional[float]
    if eval_split is None:
        if test_split_ratio_value is None:
            test_split_ratio = DEFAULT_TEST_SPLIT_RATIO
        else:
            test_split_ratio = _to_bounded_float(
                test_split_ratio_value,
                "testSplitRatio",
                min_value=MIN_TEST_SPLIT_RATIO,
                max_value=MAX_TEST_SPLIT_RATIO,
            )
    else:
        if test_split_ratio_value is not None:
            _to_bounded_float(
                test_split_ratio_value,
                "testSplitRatio",
                min_value=MIN_TEST_SPLIT_RATIO,
                max_value=MAX_TEST_SPLIT_RATIO,
            )
        test_split_ratio = None

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
        if len(experiment_name) > 120:
            raise TrainValidationError("'experimentName' must be at most 120 characters")

    _validate_algorithm_task_compatibility(task_type=task_type, training_algorithm=training_algorithm)
    if not _is_model_task_compatible(model_name=model_name.strip(), task_type=task_type):
        raise TrainValidationError(
            f"Model '{model_name.strip()}' is not compatible with taskType '{task_type}'. "
            "Pick a model that matches the selected task."
        )

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

    # Validate resource allocation params (gpu/cpu/memory)
    _validate_resource_allocation(normalized)

    return {
        "experiment_type": experiment_type,
        "model_name": model_name.strip(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "task_type": task_type,
        "training_algorithm": training_algorithm,
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split_ratio": test_split_ratio,
        "max_samples": max_samples,
        "experiment_name": experiment_name,
        "local_dataset_id": local_dataset_id,
        "local_dataset_path": local_dataset_path,
        "local_dataset_format": local_dataset_format,
        "use_gpu": normalized.get("use_gpu"),
        "gpu_count": normalized.get("gpu_count"),
        "gpu_type": normalized.get("gpu_type"),
        "cpu_cores": normalized.get("cpu_cores"),
        "memory_gb": normalized.get("memory_gb"),
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
    revision = _validate_revision(revision_value)

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
        eval_split = params_dict.get("eval_split")
        train_split = params_dict.get("train_split")
        if eval_split and eval_split != train_split:
            raise TrainValidationError(
                "For local datasets, 'evalSplit' must be empty or equal to 'trainSplit'."
            )
        dataset = f"local://pending/{local_dataset_id}"
    else:
        if not _is_non_empty_string(dataset):
            raise TrainValidationError("'dataset' is required")

    return {
        "dataset": dataset.strip(),
        "revision": revision,
        "params_dict": params_dict,
    }
