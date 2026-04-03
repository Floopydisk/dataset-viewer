# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libcommon.train import TrainValidationError, normalize_training_params, parse_training_request


def test_normalize_training_params_accepts_aliases() -> None:
    params = normalize_training_params(
        {
            "modelName": "bert-base-uncased",
            "epochs": "5",
            "batchSize": 16,
            "learningRate": "0.0005",
            "seed": "42",
        }
    )

    assert params == {
        "model_name": "bert-base-uncased",
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "seed": 42,
    }


def test_normalize_training_params_rejects_unknown_keys() -> None:
    with pytest.raises(TrainValidationError, match="Unknown training parameter"):
        normalize_training_params({"optimizer": "adam"})


def test_parse_training_request_uses_defaults() -> None:
    parsed = parse_training_request(body={"dataset": "org/ds"})

    assert parsed["dataset"] == "org/ds"
    assert parsed["revision"] == "main"
    assert parsed["params_dict"] == {
        "model_name": "tiny-bert",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "seed": None,
    }


def test_parse_training_request_rejects_dataset_mismatch() -> None:
    with pytest.raises(TrainValidationError, match="must match"):
        parse_training_request(body={"dataset": "org/one"}, dataset_query="org/two")