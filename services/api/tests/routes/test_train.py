# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from unittest.mock import patch

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
        "experiment_type": None,
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
        "local_dataset_id": None,
        "local_dataset_path": None,
        "local_dataset_format": None,
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


def test_train_post_rejects_invalid_revision_before_queueing(client: TestClient) -> None:
    with patch(
        "api.routes.train._validate_hub_dataset_revision",
        return_value="Revision 'v1' doesn't exist for dataset 'org/dataset' on the Hub.",
    ) as mock_validate_revision:
        response = client.post(
            "/train",
            json={
                "dataset": "org/dataset",
                "revision": "v1",
                "modelName": "bert-base-uncased",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Revision 'v1' doesn't exist for dataset 'org/dataset' on the Hub."
    mock_validate_revision.assert_called_once()

    queue = Queue()
    assert queue.get_dataset_pending_jobs_for_type(dataset="org/dataset", job_type="dataset-train") == []


def test_train_validate_returns_ok_for_valid_hub_revision(client: TestClient) -> None:
    with patch("api.routes.train._validate_hub_dataset_revision", return_value=None) as mock_validate_revision:
        response = client.post(
            "/train/validate",
            json={
                "datasetSource": "huggingface",
                "dataset": "org/dataset",
                "revision": "main",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["dataset"] == "org/dataset"
    assert payload["revision"] == "main"
    mock_validate_revision.assert_called_once()


def test_train_validate_rejects_invalid_hub_revision(client: TestClient) -> None:
    with patch(
        "api.routes.train._validate_hub_dataset_revision",
        return_value="Revision 'v1' doesn't exist for dataset 'org/dataset' on the Hub.",
    ):
        response = client.post(
            "/train/validate",
            json={
                "datasetSource": "huggingface",
                "dataset": "org/dataset",
                "revision": "v1",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Revision 'v1' doesn't exist for dataset 'org/dataset' on the Hub."


def test_train_validate_skips_hub_lookup_for_local_source(client: TestClient) -> None:
    with patch("api.routes.train._validate_hub_dataset_revision") as mock_validate_revision:
        response = client.post(
            "/train/validate",
            json={
                "datasetSource": "local",
                "revision": "main",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["dataset_source"] == "local"
    mock_validate_revision.assert_not_called()


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


def test_train_post_rejects_when_another_training_job_is_active(client: TestClient) -> None:
    first = client.post(
        "/train",
        json={
            "dataset": "org/dataset-a",
            "modelName": "bert-base-uncased",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/train",
        json={
            "dataset": "org/dataset-b",
            "modelName": "bert-base-uncased",
        },
    )

    assert second.status_code == 409
    payload = second.json()
    assert payload["error"] == "Another training job is already active."
    assert payload["active_job"]["dataset"] == "org/dataset-a"


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


def test_train_get_returns_live_modal_metadata_for_running_job(client: TestClient) -> None:
    post_response = client.post(
        "/train",
        json={
            "dataset": "org/dataset",
            "modelName": "bert-base-uncased",
            "trainingAlgorithm": "lora",
        },
    )
    assert post_response.status_code == 200
    job_id = post_response.json()["job_id"]

    queue = Queue()
    job_info = queue.start_job()
    assert job_info["job_id"] == job_id

    queue.update_job_params_dict(
        job_id,
        {
            "modal_run_id": "run-live-1",
            "modal_status_url": "https://modal.example.com/runs/run-live-1",
            "modal_logs_url": "https://modal.example.com/runs/run-live-1/logs",
            "modal_cancel_url": "https://modal.example.com/runs/run-live-1/cancel",
            "modal_remote_status": "running",
            "modal_remote_message": "Epoch 2/5 in progress",
            "modal_remote_updated_at": "2026-04-05T12:10:00Z",
            "structured_model_path": "models/dataset/org--dataset/revision/main/algorithm/lora/experiment/default/job/job-123/20260405T000000Z",
            "execution_backend": "modal",
            "modal_auto_shutdown": True,
        },
    )

    response = client.get("/train", params={"dataset": "org/dataset", "job_id": job_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["modal"] == {
        "modal_run_id": "run-live-1",
        "modal_status_url": "https://modal.example.com/runs/run-live-1",
        "modal_logs_url": "https://modal.example.com/runs/run-live-1/logs",
        "modal_cancel_url": "https://modal.example.com/runs/run-live-1/cancel",
        "modal_remote_status": "running",
        "modal_remote_message": "Epoch 2/5 in progress",
        "modal_remote_updated_at": "2026-04-05T12:10:00Z",
        "structured_model_path": "models/dataset/org--dataset/revision/main/algorithm/lora/experiment/default/job/job-123/20260405T000000Z",
        "execution_backend": "modal",
        "modal_auto_shutdown": True,
    }


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


def test_train_get_returns_modal_metadata_when_cache_has_artifacts(client: TestClient) -> None:
    upsert_response(
        kind="dataset-train",
        dataset="org/dataset",
        dataset_git_revision="main",
        content={
            "status": "success",
            "metrics": {"accuracy": 0.91},
            "artifacts": {
                "modal_run_id": "run-abc",
                "modal_status_url": "https://modal.example.com/runs/run-abc",
                "modal_logs_url": "https://modal.example.com/runs/run-abc/logs",
                "structured_model_path": "models/dataset/org--dataset/revision/main/algorithm/lora/experiment/default/job/job-123/20260405T000000Z",
                "execution_backend": "modal",
                "modal_auto_shutdown": True,
            },
        },
        http_status=200,
    )

    response = client.get("/train", params={"dataset": "org/dataset"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["modal"] == {
        "modal_run_id": "run-abc",
        "modal_status_url": "https://modal.example.com/runs/run-abc",
        "modal_logs_url": "https://modal.example.com/runs/run-abc/logs",
        "structured_model_path": "models/dataset/org--dataset/revision/main/algorithm/lora/experiment/default/job/job-123/20260405T000000Z",
        "execution_backend": "modal",
        "modal_auto_shutdown": True,
    }


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


def test_train_jobs_lists_active_and_ended_jobs(client: TestClient) -> None:
    queue_post = client.post(
        "/train",
        json={
            "dataset": "org/active-dataset",
            "modelName": "bert-base-uncased",
            "trainingAlgorithm": "lora",
        },
    )
    assert queue_post.status_code == 200
    job_id = queue_post.json()["job_id"]

    queue = Queue()
    queue.update_job_params_dict(
        job_id,
        {
            "modal_run_id": "run-active-1",
            "modal_status_url": "https://modal.example.com/runs/run-active-1",
            "modal_logs_url": "https://modal.example.com/runs/run-active-1/logs",
            "structured_model_path": "models/dataset/org--active-dataset/revision/main/algorithm/lora/experiment/default/job/job-123/20260405T000000Z",
            "execution_backend": "modal",
        },
    )

    upsert_response(
        kind="dataset-train",
        dataset="org/completed-dataset",
        dataset_git_revision="main",
        content={
            "status": "success",
            "model_name": "bert-base-uncased",
            "training_algorithm": "lora",
            "artifacts": {
                "structured_model_path": "https://storage.example.com/models/completed",
                "modal_run_id": "run-ended-1",
            },
        },
        http_status=HTTPStatus.OK,
    )

    response = client.get("/train/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_active"] >= 1
    assert payload["total_ended"] >= 1

    active_entry = next(item for item in payload["active"] if item["job_id"] == job_id)
    assert active_entry["dataset"] == "org/active-dataset"
    assert active_entry["status"] == "queued"
    assert active_entry["modal"]["modal_run_id"] == "run-active-1"

    ended_entry = next(item for item in payload["ended"] if item["dataset"] == "org/completed-dataset")
    assert ended_entry["status"] == "succeeded"
    assert ended_entry["model_url"] == "https://storage.example.com/models/completed"