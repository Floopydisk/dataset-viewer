# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from typing import Any

import pytest
from libcommon.dtos import Priority

from worker.config import AppConfig
from worker.job_runners.dataset.train import DatasetTrainJobRunner

from ..utils import REVISION_NAME


def _get_job_runner(app_config: AppConfig, params_dict: Mapping[str, Any]) -> DatasetTrainJobRunner:
    return DatasetTrainJobRunner(
        job_info={
            "type": DatasetTrainJobRunner.get_job_type(),
            "params": {
                "dataset": "org/dataset",
                "revision": REVISION_NAME,
                "config": None,
                "split": None,
                "params_dict": params_dict,
            },
            "job_id": "job_id",
            "priority": Priority.NORMAL,
            "difficulty": 50,
            "started_at": None,
        },
        app_config=app_config,
    )


def test_compute_uses_defaults_when_params_are_missing(app_config: AppConfig) -> None:
    job_runner = _get_job_runner(app_config=app_config, params_dict={})

    response = job_runner.compute()

    assert response.content["status"] == "success"
    assert response.content["model_name"] == "tiny-bert"
    assert response.content["epochs"] == 3
    assert response.content["batch_size"] == 32
    assert response.content["learning_rate"] == 1e-3
    assert response.content["seed"] is None
    assert response.content["task_type"] == "text-classification"
    assert response.content["training_algorithm"] is None
    assert response.content["train_split"] == "train"
    assert response.content["eval_split"] is None
    assert response.content["max_samples"] is None
    assert response.content["experiment_name"] is None
    assert response.content["artifacts"] == {}


def test_compute_normalizes_camel_case_params(app_config: AppConfig) -> None:
    job_runner = _get_job_runner(
        app_config=app_config,
        params_dict={
            "modelName": "bert-base-uncased",
            "epochs": "4",
            "batchSize": "8",
            "learningRate": "0.0002",
            "seed": "11",
            "taskType": "token-classification",
            "trainSplit": "train",
            "evalSplit": "validation",
            "maxSamples": "2000",
            "experimentName": "ner-baseline",
        },
    )

    response = job_runner.compute()

    assert response.content["status"] == "success"
    assert response.content["model_name"] == "bert-base-uncased"
    assert response.content["epochs"] == 4
    assert response.content["batch_size"] == 8
    assert response.content["learning_rate"] == 0.0002
    assert response.content["seed"] == 11
    assert response.content["task_type"] == "token-classification"
    assert response.content["train_split"] == "train"
    assert response.content["eval_split"] == "validation"
    assert response.content["max_samples"] == 2000
    assert response.content["experiment_name"] == "ner-baseline"


def test_compute_raises_for_unsupported_training_algorithm(app_config: AppConfig) -> None:
    job_runner = _get_job_runner(
        app_config=app_config,
        params_dict={
            "modelName": "bert-base-uncased",
            "trainingAlgorithm": "my-custom-algo",
        },
    )

    with pytest.raises(ValueError, match="Unsupported training algorithm"):
        job_runner.compute()