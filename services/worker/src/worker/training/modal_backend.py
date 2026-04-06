# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import json
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any
from urllib import error, request

from worker.config import ModalTrainingConfig
from worker.training.algorithms import TrainingAlgorithmResult, TrainingCancelledError, TrainingExecutionContext
from libcommon.queue.jobs import Queue  # type: ignore[import-not-found]


TERMINAL_SUCCESS_STATES = {"succeeded", "success", "completed"}
TERMINAL_CANCELLED_STATES = {"cancelled", "canceled"}
TERMINAL_FAILED_STATES = {"failed", "error"}


def _normalize_model_name(value: str) -> str:
    return value.strip().lower()


def _select_gpu_instance_class(context: TrainingExecutionContext, training_algorithm: str) -> str:
    algorithm = training_algorithm.strip().lower()
    model_name = _normalize_model_name(context["model_name"])
    task_type = str(context.get("task_type") or "").strip().lower()
    batch_size = int(context.get("batch_size") or 1)
    epochs = int(context.get("epochs") or 1)

    # Heavier fine-tuning workloads benefit from larger GPU memory/throughput.
    high_memory_model_markers = (
        "llama",
        "mistral",
        "mixtral",
        "falcon",
        "bloom",
        "gpt-j",
        "gpt-neox",
        "qwen",
        "gemma",
    )
    is_high_memory_model = any(marker in model_name for marker in high_memory_model_markers)

    if algorithm == "qlora" or (algorithm == "full-finetune" and (is_high_memory_model or batch_size >= 64)):
        return "A100"

    if algorithm in {"full-finetune", "lora"}:
        return "A10G"

    if task_type in {"causal-lm", "seq2seq", "summarization"} and (batch_size >= 32 or epochs >= 10):
        return "A10G"

    return "T4"


def _build_compute_profile(context: TrainingExecutionContext, training_algorithm: str) -> dict[str, Any]:
    gpu_instance_class = _select_gpu_instance_class(context=context, training_algorithm=training_algorithm)
    return {
        "provider": "nvidia",
        "gpu": gpu_instance_class,
        "gpu_count": 1,
    }


def _render_url_template(url_template: str, run_id: str) -> str:
    return url_template.replace("{run_id}", run_id)


def _request_json(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: int,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url=url, data=encoded, headers=dict(headers), method=method)
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as err:
        message = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Modal request failed with HTTP {err.code}: {message}") from err
    except error.URLError as err:
        raise RuntimeError(f"Modal request failed: {err.reason}") from err

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as err:
        raise RuntimeError("Modal response was not valid JSON.") from err
    if not isinstance(result, dict):
        raise RuntimeError("Modal response must be a JSON object.")
    return result


def _extract_state(result: Mapping[str, Any]) -> str:
    return str(result.get("status", "")).strip().lower()


def _request_sse_events(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    request_headers = dict(headers)
    request_headers["Accept"] = "text/event-stream"
    req = request.Request(url=url, headers=request_headers, method="GET")
    events: list[dict[str, Any]] = []

    with request.urlopen(req, timeout=timeout_seconds) as response:
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        parsed = json.loads(payload)
                        if isinstance(parsed, dict):
                            events.append(parsed)
                            state = _extract_state(parsed)
                            if state in TERMINAL_SUCCESS_STATES | TERMINAL_CANCELLED_STATES | TERMINAL_FAILED_STATES:
                                break
                    except json.JSONDecodeError:
                        continue
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

    return events


def _slugify(value: str) -> str:
    return value.strip().replace("/", "--").replace(" ", "-")


def build_structured_model_path(
    output_root: str,
    dataset: str,
    revision: str,
    training_algorithm: str,
    experiment_name: str | None,
    job_id: str,
) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    experiment_segment = _slugify(experiment_name) if experiment_name else "default"
    return (
        f"{output_root.rstrip('/')}/dataset/{_slugify(dataset)}/revision/{_slugify(revision)}/"
        f"algorithm/{_slugify(training_algorithm)}/experiment/{experiment_segment}/"
        f"job/{_slugify(job_id)}/{timestamp}"
    )


def _coerce_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    typed_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Real):
            typed_metrics[str(key)] = float(value)
    return typed_metrics


def run_training_on_modal(
    context: TrainingExecutionContext,
    modal_config: ModalTrainingConfig,
    training_algorithm: str,
) -> TrainingAlgorithmResult:
    if not modal_config.webhook_url:
        raise RuntimeError(
            "Modal training backend is enabled but MODAL_TRAINING_WEBHOOK_URL is not configured."
        )

    structured_model_path = build_structured_model_path(
        output_root=modal_config.model_output_root,
        dataset=context["dataset"],
        revision=context["revision"],
        training_algorithm=training_algorithm,
        experiment_name=context.get("experiment_name"),
        job_id=context["job_id"],
    )
    compute_profile = _build_compute_profile(context=context, training_algorithm=training_algorithm)

    payload = {
        "job_id": context["job_id"],
        "dataset": context["dataset"],
        "revision": context["revision"],
        "model_name": context["model_name"],
        "task_type": context["task_type"],
        "training_algorithm": training_algorithm,
        "train_split": context["train_split"],
        "eval_split": context.get("eval_split"),
        "epochs": context["epochs"],
        "batch_size": context["batch_size"],
        "learning_rate": context["learning_rate"],
        "seed": context.get("seed"),
        "max_samples": context.get("max_samples"),
        "experiment_name": context.get("experiment_name"),
        "local_dataset_path": context.get("local_dataset_path"),
        "local_dataset_format": context.get("local_dataset_format"),
        "output_model_path": structured_model_path,
        # Contract for ephemeral modal execution: spin up, train, persist artifacts, terminate.
        "orchestration": {
            "provider": "modal",
            "execution_mode": "ephemeral",
            "auto_shutdown": True,
            "compute": compute_profile,
        },
        "storage": {
            "kind": "s3",
            "structured_model_path": structured_model_path,
        },
    }

    headers = {"Content-Type": "application/json"}
    if modal_config.webhook_token:
        headers["Authorization"] = f"Bearer {modal_config.webhook_token}"

    result = _request_json(
        url=modal_config.webhook_url,
        headers=headers,
        timeout_seconds=modal_config.request_timeout_seconds,
        method="POST",
        payload=payload,
    )

    run_id = str(result.get("run_id", "")).strip() or None
    runtime_metadata = {
        "structured_model_path": structured_model_path,
        "execution_backend": "modal",
        "modal_auto_shutdown": True,
        "modal_gpu": compute_profile["gpu"],
        "modal_gpu_count": compute_profile["gpu_count"],
    }
    if run_id:
        runtime_metadata["modal_run_id"] = run_id
        if modal_config.logs_url_template:
            runtime_metadata["modal_logs_url"] = _render_url_template(modal_config.logs_url_template, run_id)
        if modal_config.status_url_template:
            runtime_metadata["modal_status_url"] = _render_url_template(modal_config.status_url_template, run_id)
        if modal_config.cancel_url_template:
            runtime_metadata["modal_cancel_url"] = _render_url_template(modal_config.cancel_url_template, run_id)
    Queue().update_job_params_dict(context["job_id"], runtime_metadata)

    if run_id and modal_config.status_url_template:
        status_url = _render_url_template(modal_config.status_url_template, run_id)
        events_url = f"{status_url}/events"
        logs_url = (
            _render_url_template(modal_config.logs_url_template, run_id)
            if modal_config.logs_url_template
            else None
        )
        cancel_url = (
            _render_url_template(modal_config.cancel_url_template, run_id)
            if modal_config.cancel_url_template
            else None
        )
        last_runtime_snapshot: dict[str, Any] = {}
        stream_reached_terminal_state = False
        try:
            stream_events = _request_sse_events(
                url=events_url,
                headers=headers,
                timeout_seconds=modal_config.request_timeout_seconds,
            )
            for event in stream_events:
                result = event
                state = _extract_state(result)

                runtime_snapshot: dict[str, Any] = {
                    "modal_remote_status": state or "unknown",
                }
                message = result.get("message")
                if isinstance(message, str) and message:
                    runtime_snapshot["modal_remote_message"] = message
                updated_at = result.get("updated_at")
                if isinstance(updated_at, str) and updated_at:
                    runtime_snapshot["modal_remote_updated_at"] = updated_at
                finished_at = result.get("finished_at")
                if isinstance(finished_at, str) and finished_at:
                    runtime_snapshot["modal_remote_finished_at"] = finished_at
                stage = result.get("stage")
                if isinstance(stage, str) and stage:
                    runtime_snapshot["modal_remote_stage"] = stage
                eta_seconds = result.get("training_eta_seconds")
                if isinstance(eta_seconds, (int, float)):
                    runtime_snapshot["modal_training_eta_seconds"] = float(eta_seconds)
                progress_pct = result.get("training_progress_pct")
                if isinstance(progress_pct, (int, float)):
                    runtime_snapshot["modal_training_progress_pct"] = float(progress_pct)
                submilestones = result.get("training_submilestones")
                if isinstance(submilestones, list):
                    runtime_snapshot["modal_training_submilestones"] = submilestones
                resumed_from_checkpoint = result.get("modal_resumed_from_checkpoint")
                if isinstance(resumed_from_checkpoint, bool):
                    runtime_snapshot["modal_resumed_from_checkpoint"] = resumed_from_checkpoint
                resume_checkpoint_path = result.get("modal_resume_checkpoint_path")
                if isinstance(resume_checkpoint_path, str) and resume_checkpoint_path:
                    runtime_snapshot["modal_resume_checkpoint_path"] = resume_checkpoint_path

                if runtime_snapshot != last_runtime_snapshot:
                    Queue().update_job_params_dict(context["job_id"], runtime_snapshot)
                    last_runtime_snapshot = runtime_snapshot

                cancellation_checker = context.get("cancellation_checker")
                if cancellation_checker is not None and cancellation_checker():
                    if cancel_url:
                        _request_json(
                            url=cancel_url,
                            headers=headers,
                            timeout_seconds=modal_config.request_timeout_seconds,
                            method="POST",
                            payload={"run_id": run_id},
                        )
                    raise TrainingCancelledError("Training cancelled while waiting for modal run completion.")

                if state in TERMINAL_SUCCESS_STATES | TERMINAL_CANCELLED_STATES | TERMINAL_FAILED_STATES:
                    stream_reached_terminal_state = True
                    break
        except Exception:
            stream_reached_terminal_state = False

        if not stream_reached_terminal_state:
            while True:
                cancellation_checker = context.get("cancellation_checker")
                if cancellation_checker is not None and cancellation_checker():
                    if cancel_url:
                        _request_json(
                            url=cancel_url,
                            headers=headers,
                            timeout_seconds=modal_config.request_timeout_seconds,
                            method="POST",
                            payload={"run_id": run_id},
                        )
                    raise TrainingCancelledError("Training cancelled while waiting for modal run completion.")

                result = _request_json(
                    url=status_url,
                    headers=headers,
                    timeout_seconds=modal_config.request_timeout_seconds,
                    method="GET",
                )
                state = _extract_state(result)

                runtime_snapshot = {
                    "modal_remote_status": state or "unknown",
                }
                message = result.get("message")
                if isinstance(message, str) and message:
                    runtime_snapshot["modal_remote_message"] = message
                updated_at = result.get("updated_at")
                if isinstance(updated_at, str) and updated_at:
                    runtime_snapshot["modal_remote_updated_at"] = updated_at
                finished_at = result.get("finished_at")
                if isinstance(finished_at, str) and finished_at:
                    runtime_snapshot["modal_remote_finished_at"] = finished_at
                stage = result.get("stage")
                if isinstance(stage, str) and stage:
                    runtime_snapshot["modal_remote_stage"] = stage
                eta_seconds = result.get("training_eta_seconds")
                if isinstance(eta_seconds, (int, float)):
                    runtime_snapshot["modal_training_eta_seconds"] = float(eta_seconds)
                progress_pct = result.get("training_progress_pct")
                if isinstance(progress_pct, (int, float)):
                    runtime_snapshot["modal_training_progress_pct"] = float(progress_pct)
                submilestones = result.get("training_submilestones")
                if isinstance(submilestones, list):
                    runtime_snapshot["modal_training_submilestones"] = submilestones
                resumed_from_checkpoint = result.get("modal_resumed_from_checkpoint")
                if isinstance(resumed_from_checkpoint, bool):
                    runtime_snapshot["modal_resumed_from_checkpoint"] = resumed_from_checkpoint
                resume_checkpoint_path = result.get("modal_resume_checkpoint_path")
                if isinstance(resume_checkpoint_path, str) and resume_checkpoint_path:
                    runtime_snapshot["modal_resume_checkpoint_path"] = resume_checkpoint_path

                if runtime_snapshot != last_runtime_snapshot:
                    Queue().update_job_params_dict(context["job_id"], runtime_snapshot)
                    last_runtime_snapshot = runtime_snapshot

                if state in TERMINAL_SUCCESS_STATES | TERMINAL_CANCELLED_STATES | TERMINAL_FAILED_STATES:
                    break
                time.sleep(max(1, modal_config.poll_interval_seconds))

        if logs_url:
            artifacts = result.get("artifacts")
            if isinstance(artifacts, Mapping):
                result["artifacts"] = dict(artifacts)
                result["artifacts"].setdefault("modal_logs_url", logs_url)

    status = _extract_state(result)
    if status in TERMINAL_CANCELLED_STATES:
        raise TrainingCancelledError("Training cancelled remotely by modal backend.")
    if status and status not in TERMINAL_SUCCESS_STATES:
        message = str(result.get("message", "Modal training failed."))
        raise RuntimeError(message)

    metrics_raw = result.get("metrics", {})
    artifacts_raw = result.get("artifacts", {})

    metrics = _coerce_metrics(metrics_raw if isinstance(metrics_raw, Mapping) else {})
    artifacts = dict(artifacts_raw) if isinstance(artifacts_raw, Mapping) else {}
    artifacts.setdefault("structured_model_path", structured_model_path)
    artifacts.setdefault("execution_backend", "modal")
    artifacts.setdefault("modal_auto_shutdown", True)
    artifacts.setdefault("modal_gpu", compute_profile["gpu"])
    artifacts.setdefault("modal_gpu_count", compute_profile["gpu_count"])
    artifacts.setdefault("modal_remote_status", status or "unknown")
    if isinstance(result.get("message"), str):
        artifacts.setdefault("modal_remote_message", str(result["message"]))
    if isinstance(result.get("stage"), str):
        artifacts.setdefault("modal_remote_stage", str(result["stage"]))
    if isinstance(result.get("training_eta_seconds"), (int, float)):
        artifacts.setdefault("modal_training_eta_seconds", float(result["training_eta_seconds"]))
    if isinstance(result.get("training_progress_pct"), (int, float)):
        artifacts.setdefault("modal_training_progress_pct", float(result["training_progress_pct"]))
    if isinstance(result.get("training_submilestones"), list):
        artifacts.setdefault("modal_training_submilestones", result["training_submilestones"])
    if isinstance(result.get("modal_resumed_from_checkpoint"), bool):
        artifacts.setdefault("modal_resumed_from_checkpoint", bool(result["modal_resumed_from_checkpoint"]))
    if isinstance(result.get("modal_resume_checkpoint_path"), str):
        artifacts.setdefault("modal_resume_checkpoint_path", str(result["modal_resume_checkpoint_path"]))
    if isinstance(result.get("updated_at"), str):
        artifacts.setdefault("modal_remote_updated_at", str(result["updated_at"]))
    if isinstance(result.get("finished_at"), str):
        artifacts.setdefault("modal_remote_finished_at", str(result["finished_at"]))
    if run_id:
        artifacts.setdefault("modal_run_id", run_id)
        if modal_config.logs_url_template:
            artifacts.setdefault("modal_logs_url", _render_url_template(modal_config.logs_url_template, run_id))
        if modal_config.status_url_template:
            artifacts.setdefault("modal_status_url", _render_url_template(modal_config.status_url_template, run_id))
        if modal_config.cancel_url_template:
            artifacts.setdefault("modal_cancel_url", _render_url_template(modal_config.cancel_url_template, run_id))

    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }
