# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os
from http import HTTPStatus
from collections.abc import Mapping
from typing import Any, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from mongoengine.errors import DoesNotExist

from libapi.authentication import auth_check
from libapi.utils import Endpoint, get_json_error_response, get_json_ok_response, get_cache_entry_from_step
from libcommon.config import S3Config
from libcommon.dtos import Status
from libcommon.prometheus import StepProfiler
from libcommon.queue.jobs import JobDocument, Queue
from libcommon.simple_cache import CachedResponseDocument
from libcommon.storage_client import StorageClient
from libcommon.train import TrainValidationError, get_training_capabilities, parse_training_request
from starlette.requests import Request
from starlette.responses import Response

from api.config import LocalDatasetsConfig
from api.routes.local_datasets import _create_store, _get_access_error_response, _get_request_namespace, _read_metadata


_LOCAL_DATASET_PREFIX = "local://"


def _is_local_dataset_reference(dataset: str) -> bool:
    return dataset.startswith(_LOCAL_DATASET_PREFIX)


def _build_local_dataset_reference(namespace: str, dataset_id: str) -> str:
    return f"{_LOCAL_DATASET_PREFIX}{namespace}/{dataset_id}"


def _parse_local_dataset_reference(dataset: str) -> tuple[str, str]:
    payload = dataset[len(_LOCAL_DATASET_PREFIX) :]
    namespace, dataset_id = payload.split("/", maxsplit=1)
    if not namespace or not dataset_id:
        raise ValueError("Invalid local dataset reference")
    return namespace, dataset_id


def _extract_modal_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    modal_fields = (
        "modal_run_id",
        "modal_status_url",
        "modal_logs_url",
        "modal_status_proxy_url",
        "modal_logs_proxy_url",
        "modal_cancel_url",
        "modal_remote_status",
        "modal_remote_message",
        "modal_remote_stage",
        "modal_training_eta_seconds",
        "modal_training_progress_pct",
        "modal_training_submilestones",
        "modal_remote_updated_at",
        "modal_remote_finished_at",
        "structured_model_path",
        "execution_backend",
        "modal_auto_shutdown",
        "modal_gpu",
        "modal_gpu_count",
        "modal_resumed_from_checkpoint",
        "modal_resume_checkpoint_path",
    )
    modal = {field: source[field] for field in modal_fields if field in source and source[field] is not None}
    return modal


def _append_modal_proxy_urls(modal: dict[str, Any], dataset: str) -> dict[str, Any]:
    run_id = modal.get("modal_run_id")
    if isinstance(run_id, str) and run_id:
        encoded_dataset = urlparse.quote(dataset, safe="")
        modal.setdefault("modal_status_proxy_url", f"/api/train/modal/{run_id}?dataset={encoded_dataset}")
        modal.setdefault("modal_logs_proxy_url", f"/api/train/modal/{run_id}/logs?dataset={encoded_dataset}")
    return modal


async def _authorize_train_dataset_access(
    *,
    dataset: str,
    request: Request,
    local_datasets_config: Optional[LocalDatasetsConfig],
    external_auth_url: Optional[str],
    hf_jwt_public_keys: Optional[list[str]],
    hf_jwt_algorithm: Optional[str],
    hf_timeout_seconds: Optional[float],
) -> Optional[Response]:
    if _is_local_dataset_reference(dataset):
        if local_datasets_config is None:
            return get_json_error_response(
                content={"error": "Local datasets are not configured on this deployment."},
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        access_error = _get_access_error_response(request=request, config=local_datasets_config)
        if access_error is not None:
            return access_error

        expected_namespace, _ = _parse_local_dataset_reference(dataset)
        request_namespace = _get_request_namespace(request)
        if expected_namespace != request_namespace:
            return get_json_error_response(
                content={"error": "Not authorized to access this local training job."},
                status_code=HTTPStatus.FORBIDDEN,
            )
        return None

    await auth_check(
        dataset=dataset,
        request=request,
        external_auth_url=external_auth_url,
        hf_jwt_public_keys=hf_jwt_public_keys,
        hf_jwt_algorithm=hf_jwt_algorithm,
        hf_timeout_seconds=hf_timeout_seconds,
    )
    return None


def create_train_capabilities_endpoint() -> Endpoint:
    async def train_capabilities_endpoint(request: Request) -> Response:
        with StepProfiler("train-capabilities", "endpoint"):
            if request.method != "GET":
                return get_json_error_response(
                    content={"error": "Method not allowed"}, status_code=HTTPStatus.METHOD_NOT_ALLOWED
                )
            return get_json_ok_response(content=get_training_capabilities(), max_age=0)

    return train_capabilities_endpoint


def create_train_jobs_endpoint() -> Endpoint:
    async def train_jobs_endpoint(request: Request) -> Response:
        with StepProfiler("train-jobs", "endpoint"):
            if request.method != "GET":
                return get_json_error_response(
                    content={"error": "Method not allowed"}, status_code=HTTPStatus.METHOD_NOT_ALLOWED
                )

            dataset_filter = request.query_params.get("dataset")
            limit_raw = request.query_params.get("limit", "50")
            try:
                limit = max(1, min(int(limit_raw), 200))
            except ValueError:
                return get_json_error_response(
                    content={"error": "'limit' must be an integer"}, status_code=HTTPStatus.BAD_REQUEST
                )

            active_query: dict[str, Any] = {"type": "dataset-train"}
            if dataset_filter:
                active_query["dataset"] = dataset_filter

            active_jobs = []
            for job in JobDocument.objects(**active_query).order_by("-created_at").limit(limit):
                params_dict = dict(job.params_dict or {})
                artifacts = params_dict.get("artifacts")
                artifacts_dict = dict(artifacts) if isinstance(artifacts, Mapping) else {}
                modal_source = {**params_dict, **artifacts_dict}
                active_jobs.append(
                    {
                        "job_id": str(job.pk),
                        "dataset": job.dataset,
                        "revision": job.revision,
                        "status": "running" if job.status.value == "started" else "queued",
                        "queue_status": job.status.value,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "cancel_requested": bool(job.cancel_requested),
                        "params": params_dict,
                        "modal": _append_modal_proxy_urls(_extract_modal_metadata(modal_source), job.dataset),
                        "model_name": params_dict.get("model_name"),
                        "training_algorithm": params_dict.get("training_algorithm"),
                    }
                )

            ended_query: dict[str, Any] = {"kind": "dataset-train"}
            if dataset_filter:
                ended_query["dataset"] = dataset_filter

            ended_jobs = []
            for entry in CachedResponseDocument.objects(**ended_query).order_by("-updated_at").limit(limit):
                content = dict(entry.content or {})
                artifacts = content.get("artifacts")
                artifacts_dict = dict(artifacts) if isinstance(artifacts, Mapping) else {}
                modal_source = {**content, **artifacts_dict}
                model_path = artifacts_dict.get("structured_model_path") or artifacts_dict.get("checkpoint_dir")
                model_url = model_path if isinstance(model_path, str) and model_path.startswith(("http://", "https://")) else None
                status = "succeeded" if entry.http_status == HTTPStatus.OK else "failed"
                ended_jobs.append(
                    {
                        "job_id": artifacts_dict.get("modal_run_id") or f"{entry.dataset}:{entry.updated_at.isoformat()}",
                        "dataset": entry.dataset,
                        "revision": entry.dataset_git_revision,
                        "status": status,
                        "queue_status": None,
                        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
                        "http_status": int(entry.http_status.value),
                        "error_code": entry.error_code,
                        "result": content,
                        "modal": _append_modal_proxy_urls(_extract_modal_metadata(modal_source), entry.dataset),
                        "model_name": content.get("model_name"),
                        "training_algorithm": content.get("training_algorithm"),
                        "model_path": model_path,
                        "model_url": model_url,
                    }
                )

            return get_json_ok_response(
                {
                    "active": active_jobs,
                    "ended": ended_jobs,
                    "total_active": len(active_jobs),
                    "total_ended": len(ended_jobs),
                },
                max_age=0,
            )

    return train_jobs_endpoint


def create_train_modal_proxy_endpoint(
    *,
    logs: bool,
    local_datasets_config: Optional[LocalDatasetsConfig],
    external_auth_url: Optional[str],
    hf_jwt_public_keys: Optional[list[str]],
    hf_jwt_algorithm: Optional[str],
    hf_timeout_seconds: Optional[float],
) -> Endpoint:
    async def train_modal_proxy_endpoint(request: Request) -> Response:
        with StepProfiler("train-modal-proxy", "endpoint"):
            if request.method != "GET":
                return get_json_error_response(
                    content={"error": "Method not allowed"}, status_code=HTTPStatus.METHOD_NOT_ALLOWED
                )

            dataset = request.query_params.get("dataset")
            if not dataset:
                return get_json_error_response(
                    content={"error": "'dataset' is required"}, status_code=HTTPStatus.BAD_REQUEST
                )

            access_error = await _authorize_train_dataset_access(
                dataset=dataset,
                request=request,
                local_datasets_config=local_datasets_config,
                external_auth_url=external_auth_url,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=hf_jwt_algorithm,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            if access_error is not None:
                return access_error

            run_id = request.path_params.get("run_id")
            if not run_id:
                return get_json_error_response(
                    content={"error": "'run_id' is required"}, status_code=HTTPStatus.BAD_REQUEST
                )

            template_env = "MODAL_TRAINING_LOGS_URL_TEMPLATE" if logs else "MODAL_TRAINING_STATUS_URL_TEMPLATE"
            template = os.environ.get(template_env, "")
            if not template:
                return get_json_error_response(
                    content={"error": f"{template_env} is not configured on this deployment."},
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            target_url = template.replace("{run_id}", str(run_id))
            headers: dict[str, str] = {}
            webhook_token = os.environ.get("MODAL_TRAINING_WEBHOOK_TOKEN", "")
            if webhook_token:
                headers["Authorization"] = f"Bearer {webhook_token}"

            req = urlrequest.Request(url=target_url, headers=headers, method="GET")
            try:
                with urlrequest.urlopen(req, timeout=30) as upstream:
                    payload = upstream.read()
                    content_type = upstream.headers.get("Content-Type") or (
                        "text/plain; charset=utf-8" if logs else "application/json"
                    )
                    return Response(content=payload, status_code=upstream.status, media_type=content_type)
            except urlerror.HTTPError as err:
                message = err.read().decode("utf-8", errors="replace")
                return get_json_error_response(
                    content={"error": "Modal proxy request failed.", "cause": message},
                    status_code=err.code,
                )
            except urlerror.URLError as err:
                return get_json_error_response(
                    content={"error": "Modal proxy request failed.", "cause": str(err.reason)},
                    status_code=HTTPStatus.BAD_GATEWAY,
                )

    return train_modal_proxy_endpoint


def create_train_endpoint(
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_keys: Optional[list[str]] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    storage_clients: Optional[list[StorageClient]] = None,
    local_datasets_config: Optional[LocalDatasetsConfig] = None,
    s3_config: Optional[S3Config] = None,
) -> Endpoint:
    async def train_endpoint(request: Request) -> Response:
        with StepProfiler("train", "endpoint"):
            try:
                logging.info(f"train endpoint: {request.method}")

                dataset = request.query_params.get("dataset")

                if request.method == "POST":
                    try:
                        body = await request.json()
                    except Exception:
                        return get_json_error_response(
                            content={"error": "Invalid JSON body"}, status_code=HTTPStatus.BAD_REQUEST
                        )

                    try:
                        train_request = parse_training_request(body=body, dataset_query=dataset)
                    except TrainValidationError as err:
                        return get_json_error_response(
                            content={"error": str(err)}, status_code=HTTPStatus.BAD_REQUEST
                        )

                    local_dataset_id = train_request["params_dict"].get("local_dataset_id")
                    if local_dataset_id:
                        if local_datasets_config is None or s3_config is None:
                            return get_json_error_response(
                                content={"error": "Local datasets are not configured on this deployment."},
                                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            )

                        access_error = _get_access_error_response(request=request, config=local_datasets_config)
                        if access_error is not None:
                            return access_error

                        request_namespace = _get_request_namespace(request)
                        store = _create_store(config=local_datasets_config, s3_config=s3_config)
                        metadata = _read_metadata(
                            store=store,
                            namespace=request_namespace,
                            dataset_id=local_dataset_id,
                        )
                        if metadata is None:
                            return get_json_error_response(
                                content={"error": "Local dataset not found."},
                                status_code=HTTPStatus.NOT_FOUND,
                            )

                        train_request["dataset"] = _build_local_dataset_reference(request_namespace, local_dataset_id)
                        train_request["params_dict"]["local_dataset_path"] = metadata.file_path
                        train_request["params_dict"]["local_dataset_format"] = metadata.format
                    else:
                        await auth_check(
                            dataset=train_request["dataset"],
                            request=request,
                            external_auth_url=external_auth_url,
                            hf_jwt_public_keys=hf_jwt_public_keys,
                            hf_jwt_algorithm=hf_jwt_algorithm,
                            hf_timeout_seconds=hf_timeout_seconds,
                        )

                    active_training_job = JobDocument.objects(
                        type="dataset-train", status__in=[Status.WAITING, Status.STARTED]
                    ).order_by("+created_at").first()
                    if active_training_job is not None:
                        return get_json_error_response(
                            content={
                                "error": "Another training job is already active.",
                                "cause": "Only one active training job is allowed at a time to avoid resource conflicts.",
                                "active_job": {
                                    "job_id": str(active_training_job.pk),
                                    "dataset": active_training_job.dataset,
                                    "queue_status": active_training_job.status.value,
                                },
                            },
                            status_code=HTTPStatus.CONFLICT,
                        )

                    queue = Queue()
                    job = queue.add_job(
                        job_type="dataset-train",
                        dataset=train_request["dataset"],
                        revision=train_request["revision"],
                        difficulty=50,
                        params_dict=train_request["params_dict"],
                    )

                    return get_json_ok_response(
                        {
                            "job_id": str(job.pk),
                            "status": "queued",
                            "dataset": train_request["dataset"],
                            "revision": train_request["revision"],
                            "params": train_request["params_dict"],
                            "poll": {
                                "path": "/train",
                                "params": {
                                    "dataset": train_request["dataset"],
                                    "job_id": str(job.pk),
                                },
                            },
                        },
                        max_age=0,
                    )

                elif request.method == "GET":
                    job_id = request.query_params.get("job_id")
                    if not dataset:
                        return get_json_error_response(
                            content={"error": "'dataset' is required"}, status_code=HTTPStatus.BAD_REQUEST
                        )

                    if _is_local_dataset_reference(dataset):
                        if local_datasets_config is None:
                            return get_json_error_response(
                                content={"error": "Local datasets are not configured on this deployment."},
                                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            )
                        access_error = _get_access_error_response(request=request, config=local_datasets_config)
                        if access_error is not None:
                            return access_error

                        expected_namespace, _ = _parse_local_dataset_reference(dataset)
                        request_namespace = _get_request_namespace(request)
                        if expected_namespace != request_namespace:
                            return get_json_error_response(
                                content={"error": "Not authorized to access this local training job."},
                                status_code=HTTPStatus.FORBIDDEN,
                            )
                    else:
                        await auth_check(
                            dataset=dataset,
                            request=request,
                            external_auth_url=external_auth_url,
                            hf_jwt_public_keys=hf_jwt_public_keys,
                            hf_jwt_algorithm=hf_jwt_algorithm,
                            hf_timeout_seconds=hf_timeout_seconds,
                        )

                    queue = Queue()
                    if job_id:
                        try:
                            job = queue.get_job_with_id(job_id=job_id)
                            if job.type != "dataset-train":
                                return get_json_error_response(
                                    content={"error": "job_id does not belong to a training job"},
                                    status_code=HTTPStatus.BAD_REQUEST,
                                )
                            if job.dataset != dataset:
                                return get_json_error_response(
                                    content={"error": "job_id does not belong to the requested dataset"},
                                    status_code=HTTPStatus.BAD_REQUEST,
                                )

                            state = "running" if job.status.value == "started" else "queued"
                            modal = _append_modal_proxy_urls(_extract_modal_metadata(job.params_dict), dataset)
                            return get_json_ok_response(
                                {
                                    "status": state,
                                    "job_id": str(job.pk),
                                    "dataset": dataset,
                                    "queue_status": job.status.value,
                                    **({"modal": modal} if modal else {}),
                                },
                                max_age=0,
                            )
                        except DoesNotExist:
                            # The job can already be finished and removed from the queue.
                            pass

                    try:
                        result = get_cache_entry_from_step(
                            processing_step_name="dataset-train",
                            dataset=dataset,
                            config=None,
                            split=None,
                            hf_endpoint=hf_endpoint,
                            hf_token=hf_token,
                            blocked_datasets=[],
                            hf_timeout_seconds=hf_timeout_seconds,
                            storage_clients=storage_clients,
                        )
                        if result["http_status"] != HTTPStatus.OK:
                            modal = _append_modal_proxy_urls(
                                _extract_modal_metadata(result["content"] if isinstance(result["content"], Mapping) else {}),
                                dataset,
                            )
                            return get_json_error_response(
                                content={
                                    "status": "failed",
                                    "dataset": dataset,
                                    "error_code": result["error_code"],
                                    "result": result["content"],
                                    **({"modal": modal} if modal else {}),
                                },
                                status_code=result["http_status"],
                                revision=result["dataset_git_revision"],
                            )
                        modal = _append_modal_proxy_urls(
                            _extract_modal_metadata(result["content"] if isinstance(result["content"], Mapping) else {}),
                            dataset,
                        )
                        return get_json_ok_response(
                            content={
                                "status": "succeeded",
                                "dataset": dataset,
                                "result": result["content"],
                                **({"modal": modal} if modal else {}),
                            },
                            max_age=0,
                            revision=result["dataset_git_revision"],
                        )
                    except Exception as e:
                        jobs = queue.get_dataset_pending_jobs_for_type(dataset=dataset, job_type="dataset-train")
                        if jobs:
                            has_started = any(job["status"] == "started" for job in jobs)
                            return get_json_ok_response(
                                {
                                    "status": "running" if has_started else "queued",
                                    "dataset": dataset,
                                    "jobs": jobs,
                                },
                                max_age=0,
                            )

                        logging.error(f"Error getting train status: {e}")
                        return get_json_error_response(
                            content={"status": "not_found", "error": "Training job not found or failed", "cause": str(e)},
                            status_code=HTTPStatus.NOT_FOUND,
                        )

                elif request.method == "DELETE":
                    job_id = request.query_params.get("job_id")
                    if not dataset:
                        return get_json_error_response(
                            content={"error": "'dataset' is required"}, status_code=HTTPStatus.BAD_REQUEST
                        )
                    if not job_id:
                        return get_json_error_response(
                            content={"error": "'job_id' is required"}, status_code=HTTPStatus.BAD_REQUEST
                        )

                    if _is_local_dataset_reference(dataset):
                        if local_datasets_config is None:
                            return get_json_error_response(
                                content={"error": "Local datasets are not configured on this deployment."},
                                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            )
                        access_error = _get_access_error_response(request=request, config=local_datasets_config)
                        if access_error is not None:
                            return access_error

                        expected_namespace, _ = _parse_local_dataset_reference(dataset)
                        request_namespace = _get_request_namespace(request)
                        if expected_namespace != request_namespace:
                            return get_json_error_response(
                                content={"error": "Not authorized to cancel this training job."},
                                status_code=HTTPStatus.FORBIDDEN,
                            )
                    else:
                        await auth_check(
                            dataset=dataset,
                            request=request,
                            external_auth_url=external_auth_url,
                            hf_jwt_public_keys=hf_jwt_public_keys,
                            hf_jwt_algorithm=hf_jwt_algorithm,
                            hf_timeout_seconds=hf_timeout_seconds,
                        )

                    try:
                        queue = Queue()
                        deleted_count = queue.delete_waiting_jobs_by_job_id([job_id])
                        
                        if deleted_count > 0:
                            return get_json_ok_response(
                                {
                                    "job_id": job_id,
                                    "status": "cancelled",
                                    "message": "Training job cancelled successfully",
                                },
                                max_age=0,
                            )
                        else:
                            # Job already started or doesn't exist
                            try:
                                job = queue.get_job_with_id(job_id=job_id)
                                if job.type != "dataset-train":
                                    return get_json_error_response(
                                        content={"error": "job_id does not belong to a training job"},
                                        status_code=HTTPStatus.BAD_REQUEST,
                                    )
                                if job.dataset != dataset:
                                    return get_json_error_response(
                                        content={"error": "job_id does not belong to the requested dataset"},
                                        status_code=HTTPStatus.BAD_REQUEST,
                                    )

                                if job.status.value == "started":
                                    cancellation_requested = queue.request_job_cancellation(job_id=job_id)
                                    if cancellation_requested:
                                        return get_json_ok_response(
                                            {
                                                "job_id": job_id,
                                                "status": "cancellation-requested",
                                                "message": "Cancellation requested. The running training job will stop shortly.",
                                            },
                                            max_age=0,
                                        )
                            except DoesNotExist:
                                pass
                            
                            return get_json_error_response(
                                content={
                                    "error": "Training job is no longer cancelable.",
                                    "cause": "The job may have already completed or failed. Refresh status to see the final result.",
                                },
                                status_code=HTTPStatus.NOT_FOUND,
                            )
                    except Exception as e:
                        logging.error(f"Error cancelling training job: {e}")
                        return get_json_error_response(
                            content={"error": "Failed to cancel training job", "cause": str(e)},
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )

                else:
                    return get_json_error_response(
                        content={"error": "Method not allowed"}, status_code=HTTPStatus.METHOD_NOT_ALLOWED
                    )

            except Exception as e:
                logging.error(f"Unexpected error in train endpoint: {e}")
                return get_json_error_response(
                    content={"error": "Unexpected error", "cause": str(e)},
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return train_endpoint
