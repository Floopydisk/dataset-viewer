# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from libcommon.constants import QUEUE_COLLECTION_JOBS


class TrainingChangeStreamInitError(Exception):
    pass


@dataclass
class TrainingStatusEventValue:
    job_id: str
    dataset: str
    status: str
    queue_status: str
    modal: Mapping[str, Any]
    updated_at: str


class TrainingStatusChangedEvent(asyncio.Event):
    _training_status_value: Optional[TrainingStatusEventValue]

    def __init__(self, *, training_status_value: Optional[TrainingStatusEventValue] = None):
        super().__init__()
        self._training_status_value = training_status_value
        super().set()

    def set_value(self, *, training_status_value: Optional[TrainingStatusEventValue] = None) -> None:
        self._training_status_value = training_status_value
        return super().set()

    async def wait_value(self) -> Optional[TrainingStatusEventValue]:
        await super().wait()
        return self._training_status_value


@dataclass
class TrainingSubscription:
    event: TrainingStatusChangedEvent
    dataset: str
    job_id: Optional[str]


@dataclass
class TrainingPublisher:
    _watchers: dict[str, TrainingSubscription]

    def _notify_change(self, value: TrainingStatusEventValue) -> None:
        for watcher in self._watchers.values():
            if watcher.dataset != value.dataset:
                continue
            if watcher.job_id is not None and watcher.job_id != value.job_id:
                continue
            watcher.event.set_value(training_status_value=value)

    def _unsubscribe(self, uuid: str) -> None:
        self._watchers.pop(uuid, None)

    def _subscribe(self, *, dataset: str, job_id: Optional[str]) -> tuple[str, TrainingStatusChangedEvent]:
        event = TrainingStatusChangedEvent()
        uuid = uuid4().hex
        self._watchers[uuid] = TrainingSubscription(event=event, dataset=dataset, job_id=job_id)
        return (uuid, event)


def _extract_modal_metadata(params_dict: Mapping[str, Any]) -> dict[str, Any]:
    modal_fields = (
        "modal_run_id",
        "modal_status_url",
        "modal_logs_url",
        "modal_cancel_url",
        "modal_remote_status",
        "modal_remote_message",
        "modal_remote_updated_at",
        "modal_remote_finished_at",
        "structured_model_path",
        "execution_backend",
        "modal_auto_shutdown",
    )
    return {field: params_dict[field] for field in modal_fields if field in params_dict and params_dict[field] is not None}


def _queue_status_to_status(queue_status: str) -> str:
    if queue_status == "started":
        return "running"
    if queue_status == "waiting":
        return "queued"
    return queue_status


def _modal_remote_status_to_status(modal_remote_status: str | None) -> str | None:
    if modal_remote_status is None:
        return None
    normalized = modal_remote_status.strip().lower()
    if normalized in TERMINAL_SUCCESS_STATES:
        return "succeeded"
    if normalized in TERMINAL_CANCELLED_STATES:
        return "cancelled"
    if normalized in TERMINAL_FAILED_STATES:
        return "failed"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_event_value(document: Mapping[str, Any]) -> Optional[TrainingStatusEventValue]:
    if str(document.get("type", "")) != "dataset-train":
        return None
    job_id_value = document.get("_id")
    if not isinstance(job_id_value, ObjectId):
        return None
    dataset = document.get("dataset")
    queue_status = document.get("status")
    if not isinstance(dataset, str) or not isinstance(queue_status, str):
        return None

    params_dict = document.get("params_dict")
    modal = _extract_modal_metadata(params_dict if isinstance(params_dict, Mapping) else {})
    modal_remote_status = modal.get("modal_remote_status")
    effective_status = _modal_remote_status_to_status(str(modal_remote_status) if modal_remote_status else None)

    return TrainingStatusEventValue(
        job_id=str(job_id_value),
        dataset=dataset,
        status=effective_status or _queue_status_to_status(queue_status),
        queue_status=queue_status,
        modal=modal,
        updated_at=_now(),
    )


class TrainingWatcher:
    _watch_task: asyncio.Task[None]

    def __init__(self, client: AsyncIOMotorClient, db_name: str) -> None:
        self._client = client
        self._collection = self._client[db_name][QUEUE_COLLECTION_JOBS]
        self._publisher = TrainingPublisher(_watchers={})

    def start_watching(self) -> None:
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watching(self) -> None:
        self._watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._watch_task

    def subscribe(self, *, dataset: str, job_id: Optional[str]) -> tuple[str, TrainingStatusChangedEvent]:
        return self._publisher._subscribe(dataset=dataset, job_id=job_id)

    def unsubscribe(self, uuid: str) -> None:
        self._publisher._unsubscribe(uuid)

    async def get_initial_value(self, *, dataset: str, job_id: Optional[str]) -> Optional[TrainingStatusEventValue]:
        query: dict[str, Any] = {"type": "dataset-train", "dataset": dataset}
        if job_id:
            try:
                query["_id"] = ObjectId(job_id)
            except Exception:
                return None
            document = await self._collection.find_one(query)
        else:
            document = await self._collection.find_one(query, sort=[("created_at", -1)])

        if document is None:
            return None
        return _to_event_value(document)

    async def _watch_loop(self) -> None:
        pipeline: Sequence[Mapping[str, Any]] = [
            {
                "$match": {
                    "fullDocument.type": "dataset-train",
                    "operationType": {"$in": ["insert", "update", "replace"]},
                },
            },
            {
                "$project": {
                    "fullDocument": 1,
                    "updateDescription": 1,
                    "operationType": 1,
                },
            },
        ]
        resume_token = None
        while True:
            try:
                async with self._collection.watch(
                    pipeline,
                    resume_after=resume_token,
                    full_document="updateLookup",
                ) as stream:
                    async for change in stream:
                        resume_token = stream.resume_token
                        operation = change.get("operationType")
                        if operation == "update":
                            updated_fields = change.get("updateDescription", {}).get("updatedFields", {})
                            if not any(
                                field.startswith("status")
                                or field.startswith("params_dict")
                                or field.startswith("cancel_requested")
                                or field.startswith("last_heartbeat")
                                for field in updated_fields
                            ):
                                continue

                        full_document = change.get("fullDocument")
                        if not isinstance(full_document, Mapping):
                            continue

                        value = _to_event_value(full_document)
                        if value is None:
                            continue
                        self._publisher._notify_change(value)
            except PyMongoError:
                if resume_token is None:
                    raise TrainingChangeStreamInitError()
