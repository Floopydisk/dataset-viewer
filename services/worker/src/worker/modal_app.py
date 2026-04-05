# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from __future__ import annotations

import os
import uuid
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import modal

def _resolve_worker_src_dir() -> Path:
    configured = os.environ.get("WORKER_SRC_DIR")
    if configured:
        configured_path = Path(configured).resolve()
        if configured_path.exists():
            return configured_path

    current_file = Path(__file__).resolve()
    search_roots = [current_file.parent, *list(current_file.parents)]
    for root in search_roots:
        candidate = (root / "services" / "worker" / "src").resolve()
        if candidate.exists():
            return candidate

    # Modal runtime fallback when file is imported from /root/modal_app.py.
    modal_fallback = Path("/root/services/worker/src")
    if modal_fallback.exists():
        return modal_fallback

    raise RuntimeError("Unable to locate services/worker/src for modal training runtime.")


WORKER_SRC_DIR = _resolve_worker_src_dir()
for source_dir in (str(WORKER_SRC_DIR),):
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

RUN_STATE_DICT_NAME = "dataset-viewer-modal-training-runs"
WEBHOOK_SECRET_NAME = "dataset-viewer-training"
WEBHOOK_TOKEN_ENV_VAR = "WEBHOOK_TOKEN"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "starlette==0.37.1",
        "numpy",
        "transformers==4.41.0",
        "accelerate>=1.13.0",
        "peft==0.11.0",
        "datasets",
        "torch",
    )
    .env({"PYTHONPATH": "/root/services/worker/src"})
    .add_local_dir(str(WORKER_SRC_DIR), remote_path="/root/services/worker/src")
)

app = modal.App("dataset-viewer-training")
RUNS = modal.Dict.from_name(RUN_STATE_DICT_NAME, create_if_missing=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_url(request: Any) -> str:
    return str(request.base_url).rstrip("/")


def _require_webhook_token(request: Any) -> None:
    expected_token = os.environ.get(WEBHOOK_TOKEN_ENV_VAR, "")
    if not expected_token:
        return
    authorization = request.headers.get("authorization", "")
    if authorization != f"Bearer {expected_token}":
        raise RuntimeError("Unauthorized")


def _structured_model_path(payload: Mapping[str, Any]) -> str:
    def _slugify(value: str) -> str:
        return value.strip().replace("/", "--").replace(" ", "-")

    output_root = str(payload.get("output_root") or "models")
    dataset = str(payload["dataset"])
    revision = str(payload["revision"])
    training_algorithm = str(payload.get("training_algorithm") or "full-finetune")
    experiment_name = payload.get("experiment_name")
    experiment_segment = _slugify(str(experiment_name)) if experiment_name else "default"
    job_id = str(payload["job_id"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return (
        f"{output_root.rstrip('/')}/dataset/{_slugify(dataset)}/revision/{_slugify(revision)}/"
        f"algorithm/{_slugify(training_algorithm)}/experiment/{experiment_segment}/"
        f"job/{_slugify(job_id)}/{timestamp}"
    )


def _run_artifacts(base_url: str, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    structured_model_path = _structured_model_path(payload)
    return {
        "structured_model_path": structured_model_path,
        "execution_backend": "modal",
        "modal_auto_shutdown": True,
        "modal_run_id": run_id,
        "modal_status_url": f"{base_url}/runs/{run_id}",
        "modal_logs_url": f"{base_url}/runs/{run_id}/logs",
        "modal_cancel_url": f"{base_url}/runs/{run_id}/cancel",
    }


def _initial_state(base_url: str, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = _run_artifacts(base_url, run_id, payload)
    return {
        "run_id": run_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "payload": dict(payload),
        "metrics": {},
        "artifacts": artifacts,
        "logs": [f"Run {run_id} queued on Modal."],
        "cancel_requested": False,
        "message": "Training run queued.",
    }


def _get_state(run_id: str) -> dict[str, Any] | None:
    state = RUNS.get(run_id)
    if state is None:
        return None
    return dict(state)


def _store_state(run_id: str, state: Mapping[str, Any]) -> None:
    RUNS.put(run_id, dict(state))


def _append_log(run_id: str, message: str) -> None:
    state = _get_state(run_id)
    if state is None:
        return
    logs = list(state.get("logs") or [])
    logs.append(f"{_now()} {message}")
    state["logs"] = logs
    state["updated_at"] = _now()
    _store_state(run_id, state)


def _set_status(run_id: str, status: str, message: str | None = None, **updates: Any) -> None:
    state = _get_state(run_id)
    if state is None:
        return
    state.update(updates)
    state["status"] = status
    state["updated_at"] = _now()
    if message is not None:
        state["message"] = message
    _store_state(run_id, state)


def _is_cancel_requested(run_id: str) -> bool:
    state = _get_state(run_id)
    return bool(state and state.get("cancel_requested"))


def _make_training_context(run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(payload["job_id"]),
        "dataset": str(payload["dataset"]),
        "revision": str(payload["revision"]),
        "model_name": str(payload["model_name"]),
        "task_type": str(payload["task_type"]),
        "train_split": str(payload["train_split"]),
        "eval_split": payload.get("eval_split"),
        "epochs": int(payload["epochs"]),
        "batch_size": int(payload["batch_size"]),
        "learning_rate": float(payload["learning_rate"]),
        "seed": payload.get("seed"),
        "max_samples": payload.get("max_samples"),
        "experiment_name": payload.get("experiment_name"),
        "local_dataset_path": payload.get("local_dataset_path"),
        "local_dataset_format": payload.get("local_dataset_format"),
        "cancellation_checker": lambda: _is_cancel_requested(run_id),
    }


@app.function(image=image, timeout=60 * 60 * 8)
def train_remote(run_id: str, payload: dict[str, Any]) -> None:
    from worker.training.algorithms import TrainingCancelledError, TrainingExecutionContext, run_training_algorithm

    base_url = str(payload.get("base_url") or "").rstrip("/")
    state = _get_state(run_id)
    if state is None:
        state = _initial_state(base_url=base_url, run_id=run_id, payload=payload)
    _set_status(run_id, "running", "Training started.")
    _append_log(run_id, "Training job started.")
    _append_log(run_id, f"Dataset: {payload.get('dataset')}")
    _append_log(run_id, f"Algorithm: {payload.get('training_algorithm')}")

    try:
        normalized = dict(payload)
        context = cast(TrainingExecutionContext, _make_training_context(run_id, normalized))
        training_algorithm = str(normalized.get("training_algorithm") or "full-finetune")
        _append_log(run_id, f"Structured model path: {_structured_model_path(normalized)}")
        algorithm_result = run_training_algorithm(name=training_algorithm, context=context)
        metrics = dict(algorithm_result["metrics"])
        artifacts = dict(algorithm_result["artifacts"])
        artifacts.update(_run_artifacts(base_url, run_id, normalized))
        _set_status(
            run_id,
            "succeeded",
            "Training finished successfully.",
            metrics=metrics,
            artifacts=artifacts,
            finished_at=_now(),
        )
        _append_log(run_id, "Training completed successfully.")
    except TrainingCancelledError:
        _set_status(run_id, "cancelled", "Training was cancelled.", finished_at=_now())
        _append_log(run_id, "Training cancelled.")
    except Exception as err:  # pragma: no cover - runtime path
        _set_status(
            run_id,
            "failed",
            f"Training failed: {type(err).__name__}",
            error_type=type(err).__name__,
            error_message=str(err),
            finished_at=_now(),
        )
        _append_log(run_id, f"Training failed: {type(err).__name__}: {err}")
        raise


async def _train(request: Any) -> Any:
    from starlette.responses import JSONResponse

    try:
        _require_webhook_token(request)
    except RuntimeError:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

    if "job_id" not in payload or "dataset" not in payload or "revision" not in payload:
        return JSONResponse({"error": "job_id, dataset and revision are required"}, status_code=400)

    run_id = uuid.uuid4().hex
    payload = dict(payload)
    payload["base_url"] = _base_url(request)
    _store_state(run_id, _initial_state(payload["base_url"], run_id, payload))

    try:
        train_remote.spawn(run_id, payload)
    except Exception as err:
        _set_status(run_id, "failed", f"Failed to spawn remote job: {type(err).__name__}")
        _append_log(run_id, f"Spawn failed: {type(err).__name__}: {err}")
        return JSONResponse(
            {
                "run_id": run_id,
                "status": "failed",
                "message": f"Failed to start training: {type(err).__name__}",
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "run_id": run_id,
            "status": "queued",
            "message": "Training job submitted to Modal.",
            "artifacts": _run_artifacts(payload["base_url"], run_id, payload),
        }
    )


async def _status(request: Any) -> Any:
    from starlette.responses import JSONResponse

    try:
        _require_webhook_token(request)
    except RuntimeError:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    run_id = request.path_params["run_id"]
    state = _get_state(run_id)
    if state is None:
        return JSONResponse({"error": "Run not found", "status": "not_found"}, status_code=404)
    return JSONResponse(state)


async def _cancel(request: Any) -> Any:
    from starlette.responses import JSONResponse

    try:
        _require_webhook_token(request)
    except RuntimeError:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    run_id = request.path_params["run_id"]
    state = _get_state(run_id)
    if state is None:
        return JSONResponse({"error": "Run not found", "status": "not_found"}, status_code=404)
    state["cancel_requested"] = True
    state["status"] = "cancellation-requested"
    state["message"] = "Cancellation requested. The remote job will stop shortly."
    state["updated_at"] = _now()
    _store_state(run_id, state)
    _append_log(run_id, "Cancellation requested.")
    return JSONResponse({"run_id": run_id, "status": "cancellation-requested", "message": state["message"]})


async def _logs(request: Any) -> Any:
    from starlette.responses import JSONResponse, PlainTextResponse

    try:
        _require_webhook_token(request)
    except RuntimeError:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    run_id = request.path_params["run_id"]
    state = _get_state(run_id)
    if state is None:
        return JSONResponse({"error": "Run not found", "status": "not_found"}, status_code=404)
    logs = state.get("logs") or []
    return PlainTextResponse("\n".join(str(line) for line in logs))


@app.function(image=image, secrets=[modal.Secret.from_name(WEBHOOK_SECRET_NAME, required_keys=[WEBHOOK_TOKEN_ENV_VAR])])
@modal.asgi_app()
def modal_training_api() -> Any:
    from starlette.applications import Starlette
    from starlette.routing import Route

    return Starlette(
        routes=[
            Route("/train", _train, methods=["POST"]),
            Route("/runs/{run_id}", _status, methods=["GET"]),
            Route("/runs/{run_id}/cancel", _cancel, methods=["POST"]),
            Route("/runs/{run_id}/logs", _logs, methods=["GET"]),
        ]
    )


@app.local_entrypoint()
def main() -> None:
    print("Deploy with: modal deploy services/worker/src/worker/modal_app.py")
