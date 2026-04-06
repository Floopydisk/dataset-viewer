# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from __future__ import annotations

import asyncio
import json
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
CHECKPOINT_VOLUME_NAME = "dataset-viewer-training-checkpoints"
CHECKPOINT_VOLUME_MOUNT_PATH = "/vol/checkpoints"
TRAINING_OUTPUT_ROOT_ENV_VAR = "TRAINING_OUTPUT_ROOT"

CHECKPOINT_VOLUME = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "starlette==0.37.1",
        "numpy",
        "transformers==4.41.0",
        "accelerate>=1.13.0",
        "peft==0.11.0",
        "datasets",
        "bitsandbytes",
    )
    .run_commands("python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128")
    .env({"PYTHONPATH": "/root/services/worker/src", TRAINING_OUTPUT_ROOT_ENV_VAR: CHECKPOINT_VOLUME_MOUNT_PATH})
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
        "modal_checkpoint_volume": CHECKPOINT_VOLUME_NAME,
        "modal_checkpoint_mount_path": CHECKPOINT_VOLUME_MOUNT_PATH,
    }


def _initial_state(base_url: str, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = _run_artifacts(base_url, run_id, payload)
    gpu_class = _resolve_gpu_class(payload)
    artifacts.setdefault("modal_gpu", gpu_class)
    artifacts.setdefault("modal_gpu_count", 1)
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
        "stage": "queued",
        "gpu_class": gpu_class,
        "training_eta_seconds": None,
        "training_progress_pct": 0.0,
        "training_submilestones": [
            {"key": "trainer-started", "label": "Trainer started", "status": "pending"},
            {"key": "first-steps", "label": "First batches complete", "status": "pending"},
            {"key": "midpoint", "label": "Training midpoint", "status": "pending"},
            {"key": "final-steps", "label": "Final optimization steps", "status": "pending"},
            {"key": "train-loop-complete", "label": "Train loop complete", "status": "pending"},
        ],
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


def _set_stage(run_id: str, stage: str, message: str) -> None:
    _set_status(run_id, "running", message, stage=stage)
    _append_log(run_id, f"[{stage}] {message}")


def _update_submilestones(
    current: list[dict[str, Any]],
    *,
    progress_pct: float,
    train_started: bool,
    train_ended: bool,
) -> list[dict[str, Any]]:
    next_items = [dict(item) for item in current]
    status_by_key: dict[str, str] = {
        "trainer-started": "completed" if train_started or progress_pct > 0 else "pending",
        "first-steps": "completed" if progress_pct >= 5 else ("in-progress" if progress_pct > 0 else "pending"),
        "midpoint": "completed" if progress_pct >= 50 else ("in-progress" if progress_pct >= 20 else "pending"),
        "final-steps": "completed" if progress_pct >= 90 else ("in-progress" if progress_pct >= 70 else "pending"),
        "train-loop-complete": "completed" if train_ended or progress_pct >= 100 else "pending",
    }
    for item in next_items:
        key = str(item.get("key") or "")
        status = status_by_key.get(key)
        if status is not None:
            item["status"] = status
            if status == "completed" and item.get("completed_at") is None:
                item["completed_at"] = _now()
    return next_items


def _training_progress_callback(run_id: str):
    def _callback(payload: Mapping[str, Any]) -> None:
        state = _get_state(run_id)
        if state is None:
            return

        event = str(payload.get("event") or "")
        progress_pct_raw = payload.get("progress_pct")
        eta_seconds_raw = payload.get("eta_seconds")
        progress_pct = float(progress_pct_raw) if isinstance(progress_pct_raw, (int, float)) else 0.0
        if progress_pct < 0:
            progress_pct = 0.0
        if progress_pct > 100:
            progress_pct = 100.0

        eta_seconds = None
        if isinstance(eta_seconds_raw, (int, float)):
            eta_seconds = max(0.0, float(eta_seconds_raw))

        submilestones = state.get("training_submilestones")
        if not isinstance(submilestones, list):
            submilestones = []
        state["training_submilestones"] = _update_submilestones(
            [item for item in submilestones if isinstance(item, Mapping)],
            progress_pct=progress_pct,
            train_started=event == "train_begin",
            train_ended=event == "train_end",
        )
        state["training_progress_pct"] = progress_pct
        state["training_eta_seconds"] = eta_seconds
        state["updated_at"] = _now()
        _store_state(run_id, state)

    return _callback


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
        "progress_callback": _training_progress_callback(run_id),
    }


def _execute_training(run_id: str, payload: dict[str, Any]) -> None:
    from worker.training.algorithms import TrainingCancelledError, TrainingExecutionContext, run_training_algorithm

    base_url = str(payload.get("base_url") or "").rstrip("/")
    state = _get_state(run_id)
    if state is None:
        state = _initial_state(base_url=base_url, run_id=run_id, payload=payload)
    _set_status(run_id, "running", "Training started.", stage="initializing")
    _append_log(run_id, "Training job started.")
    _append_log(run_id, f"Dataset: {payload.get('dataset')}")
    _append_log(run_id, f"Algorithm: {payload.get('training_algorithm')}")
    _append_log(run_id, f"GPU class: {_resolve_gpu_class(payload)}")

    try:
        normalized = dict(payload)
        context = cast(TrainingExecutionContext, _make_training_context(run_id, normalized))
        training_algorithm = str(normalized.get("training_algorithm") or "full-finetune")
        _set_stage(run_id, "preparing_data", "Preparing dataset and training inputs.")
        _append_log(run_id, f"Structured model path: {_structured_model_path(normalized)}")
        _set_stage(run_id, "training", "Model training in progress. This can take several minutes.")
        algorithm_result = run_training_algorithm(name=training_algorithm, context=context)
        _set_stage(run_id, "saving_artifacts", "Training finished. Saving artifacts and metrics.")
        metrics = dict(algorithm_result["metrics"])
        artifacts = dict(algorithm_result["artifacts"])
        state_after_training = _get_state(run_id) or {}
        if isinstance(state_after_training.get("training_submilestones"), list):
            artifacts["modal_training_submilestones"] = state_after_training["training_submilestones"]
        if isinstance(state_after_training.get("training_eta_seconds"), (int, float)):
            artifacts["modal_training_eta_seconds"] = float(state_after_training["training_eta_seconds"])
        if isinstance(state_after_training.get("training_progress_pct"), (int, float)):
            artifacts["modal_training_progress_pct"] = float(state_after_training["training_progress_pct"])

        resumed_from_checkpoint = artifacts.get("modal_resumed_from_checkpoint")
        resume_checkpoint_path = artifacts.get("modal_resume_checkpoint_path")

        artifacts.update(_run_artifacts(base_url, run_id, normalized))

        status_updates: dict[str, Any] = {
            "metrics": metrics,
            "artifacts": artifacts,
            "finished_at": _now(),
            "stage": "completed",
            "training_eta_seconds": 0.0,
            "training_progress_pct": 100.0,
        }
        if isinstance(resumed_from_checkpoint, bool):
            status_updates["modal_resumed_from_checkpoint"] = resumed_from_checkpoint
        if isinstance(resume_checkpoint_path, str) and resume_checkpoint_path:
            status_updates["modal_resume_checkpoint_path"] = resume_checkpoint_path

        _set_status(
            run_id,
            "succeeded",
            "Training finished successfully.",
            **status_updates,
        )
        _append_log(run_id, "Training completed successfully.")
    except TrainingCancelledError:
        _set_status(run_id, "cancelled", "Training was cancelled.", finished_at=_now(), stage="cancelled")
        _append_log(run_id, "Training cancelled.")
    except Exception as err:  # pragma: no cover - runtime path
        _set_status(
            run_id,
            "failed",
            f"Training failed: {type(err).__name__}",
            error_type=type(err).__name__,
            error_message=str(err),
            finished_at=_now(),
            stage="failed",
        )
        _append_log(run_id, f"Training failed: {type(err).__name__}: {err}")
        raise


@app.function(image=image, timeout=60 * 60 * 8, gpu="T4", volumes={CHECKPOINT_VOLUME_MOUNT_PATH: CHECKPOINT_VOLUME})
def train_remote_t4(run_id: str, payload: dict[str, Any]) -> None:
    _execute_training(run_id=run_id, payload=payload)


@app.function(image=image, timeout=60 * 60 * 8, gpu="A10G", volumes={CHECKPOINT_VOLUME_MOUNT_PATH: CHECKPOINT_VOLUME})
def train_remote_a10g(run_id: str, payload: dict[str, Any]) -> None:
    _execute_training(run_id=run_id, payload=payload)


@app.function(image=image, timeout=60 * 60 * 8, gpu="A100", volumes={CHECKPOINT_VOLUME_MOUNT_PATH: CHECKPOINT_VOLUME})
def train_remote_a100(run_id: str, payload: dict[str, Any]) -> None:
    _execute_training(run_id=run_id, payload=payload)


def _resolve_gpu_class(payload: Mapping[str, Any]) -> str:
    orchestration = payload.get("orchestration")
    if isinstance(orchestration, Mapping):
        compute = orchestration.get("compute")
        if isinstance(compute, Mapping):
            gpu = compute.get("gpu")
            if isinstance(gpu, str) and gpu.strip():
                return gpu.strip().upper()
    return "A10G"


def _spawn_remote_training(run_id: str, payload: dict[str, Any]) -> None:
    gpu_class = _resolve_gpu_class(payload)
    if gpu_class == "A100":
        train_remote_a100.spawn(run_id, payload)
        return
    if gpu_class == "T4":
        train_remote_t4.spawn(run_id, payload)
        return
    train_remote_a10g.spawn(run_id, payload)


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
        _spawn_remote_training(run_id, payload)
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


async def _events(request: Any) -> Any:
    from starlette.responses import JSONResponse, StreamingResponse

    try:
        _require_webhook_token(request)
    except RuntimeError:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    run_id = request.path_params["run_id"]
    state = _get_state(run_id)
    if state is None:
        return JSONResponse({"error": "Run not found", "status": "not_found"}, status_code=404)

    async def generator() -> Any:
        last_updated_at: str | None = None
        while True:
            current_state = _get_state(run_id)
            if current_state is None:
                break

            updated_at = str(current_state.get("updated_at") or "")
            if updated_at != last_updated_at:
                last_updated_at = updated_at
                yield f"data: {json.dumps(current_state)}\n\n"
            else:
                # keep-alive frame for long-running steps so clients don't time out.
                yield ": keep-alive\n\n"

            if str(current_state.get("status", "")).lower() in {
                "succeeded",
                "success",
                "completed",
                "failed",
                "error",
                "cancelled",
                "canceled",
            }:
                break

            if await request.is_disconnected():
                break

            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.function(image=image, secrets=[modal.Secret.from_name(WEBHOOK_SECRET_NAME, required_keys=[WEBHOOK_TOKEN_ENV_VAR])])
@modal.asgi_app()
def modal_training_api() -> Any:
    from starlette.applications import Starlette
    from starlette.routing import Route

    return Starlette(
        routes=[
            Route("/train", _train, methods=["POST"]),
            Route("/runs/{run_id}", _status, methods=["GET"]),
            Route("/runs/{run_id}/events", _events, methods=["GET"]),
            Route("/runs/{run_id}/cancel", _cancel, methods=["POST"]),
            Route("/runs/{run_id}/logs", _logs, methods=["GET"]),
        ]
    )


@app.local_entrypoint()
def main() -> None:
    print("Deploy with: modal deploy services/worker/src/worker/modal_app.py")
