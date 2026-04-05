# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from mongoengine.errors import DoesNotExist

from libapi.authentication import auth_check
from libapi.utils import Endpoint, get_json_error_response, get_json_ok_response, get_cache_entry_from_step
from libcommon.config import S3Config
from libcommon.prometheus import StepProfiler
from libcommon.queue.jobs import Queue
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


def create_train_capabilities_endpoint() -> Endpoint:
    async def train_capabilities_endpoint(request: Request) -> Response:
        with StepProfiler("train-capabilities", "endpoint"):
            if request.method != "GET":
                return get_json_error_response(
                    content={"error": "Method not allowed"}, status_code=HTTPStatus.METHOD_NOT_ALLOWED
                )
            return get_json_ok_response(content=get_training_capabilities(), max_age=0)

    return train_capabilities_endpoint


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
                            return get_json_ok_response(
                                {
                                    "status": state,
                                    "job_id": str(job.pk),
                                    "dataset": dataset,
                                    "queue_status": job.status.value,
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
                            return get_json_error_response(
                                content={
                                    "status": "failed",
                                    "dataset": dataset,
                                    "error_code": result["error_code"],
                                    "result": result["content"],
                                },
                                status_code=result["http_status"],
                                revision=result["dataset_git_revision"],
                            )
                        return get_json_ok_response(
                            content={
                                "status": "succeeded",
                                "dataset": dataset,
                                "result": result["content"],
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
