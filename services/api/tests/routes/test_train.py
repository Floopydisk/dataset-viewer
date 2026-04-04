# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus

from libcommon.queue.jobs import Queue
from libcommon.simple_cache import upsert_response
from starlette.testclient import TestClient


def test_train_post_queues_job_with_normalized_params(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "revision": "main",
            "modelName": "bert-base-uncased",
            "epochs": "4",
            "batchSize": 8,
            "learningRate": "0.0002",
            "seed": "123",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["dataset"] == "org/dataset"
    assert payload["poll"]["path"] == "/train"
    assert payload["poll"]["params"]["dataset"] == "org/dataset"
    assert payload["poll"]["params"]["job_id"] == payload["job_id"]
    assert payload["params"] == {
        "model_name": "bert-base-uncased",
        "epochs": 4,
        "batch_size": 8,
        "learning_rate": 0.0002,
        "seed": 123,
        "task_type": "text-classification",
        "training_algorithm": None,
        "train_split": "train",
        "eval_split": None,
        "max_samples": None,
        "experiment_name": None,
    }

    queue = Queue()
    job_info = queue.start_job()
    assert job_info["type"] == "dataset-train"
    assert job_info["params"]["dataset"] == "org/dataset"
    assert job_info["params"]["revision"] == "main"
    assert job_info["params"]["params_dict"] == payload["params"]


def test_train_post_rejects_missing_dataset(client: TestClient) -> None:
    response = client.post("/train", json={"modelName": "bert-base-uncased"})

    assert response.status_code == 400
    assert response.json()["error"] == "'dataset' is required"


def test_train_post_rejects_invalid_hyperparameter(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
            "epochs": 0,
        },
    )

    assert response.status_code == 400
    assert "epochs" in response.json()["error"]


def test_train_post_rejects_unknown_hyperparameter(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
            "optimizer": "adamw",
        },
    )

    assert response.status_code == 400
    assert "Unknown training parameter" in response.json()["error"]


def test_train_get_requires_dataset(client: TestClient) -> None:
    response = client.get("/train")

    assert response.status_code == 400
    assert response.json()["error"] == "'dataset' is required"


def test_train_get_returns_job_state_by_job_id(client: TestClient) -> None:
    post_response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
        },
    )
    assert post_response.status_code == 200
    job_id = post_response.json()["job_id"]

    queued_response = client.get("/train", params={"dataset": "org/dataset", "job_id": job_id})
    assert queued_response.status_code == 200
    assert queued_response.json()["status"] == "queued"

    queue = Queue()
    queue.start_job()

    running_response = client.get("/train", params={"dataset": "org/dataset", "job_id": job_id})
    assert running_response.status_code == 200
    assert running_response.json()["status"] == "running"


def test_train_get_returns_succeeded_when_cache_is_available(client: TestClient) -> None:
    upsert_response(
        kind="dataset-train",
        dataset="org/dataset",
        dataset_git_revision="main",
        content={"status": "success", "accuracy": 0.91},
        http_status=200,
    )

    response = client.get("/train", params={"dataset": "org/dataset"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["result"] == {"status": "success", "accuracy": 0.91}


def test_train_get_returns_failed_when_cache_is_an_error(client: TestClient) -> None:
    upsert_response(
        kind="dataset-train",
        dataset="org/dataset",
        dataset_git_revision="main",
        content={"error": "training failed"},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code="TrainingError",
    )

    response = client.get("/train", params={"dataset": "org/dataset"})

    assert response.status_code == 500
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error_code"] == "TrainingError"
    assert payload["result"] == {"error": "training failed"}


def test_train_capabilities_returns_supported_values(client: TestClient) -> None:
    response = client.get("/train/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert "task_types" in payload
    assert "training_algorithms" in payload
    assert "text-classification" in payload["task_types"]
    assert "full-finetune" in payload["training_algorithms"]


def test_train_post_rejects_unsupported_task_type(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
            "taskType": "image-classification",
        },
    )

    assert response.status_code == 400
    assert "Unsupported taskType" in response.json()["error"]


def test_train_post_rejects_unsupported_training_algorithm(client: TestClient) -> None:
    response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
            "trainingAlgorithm": "my-custom-algo",
        },
    )

    assert response.status_code == 400
    assert "Unsupported trainingAlgorithm" in response.json()["error"]