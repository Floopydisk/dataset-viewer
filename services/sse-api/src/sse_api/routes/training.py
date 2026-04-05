# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import dataclasses
import logging

from starlette.websockets import WebSocket, WebSocketDisconnect

from sse_api.training_watcher import TrainingWatcher


def create_train_status_websocket_endpoint(training_watcher: TrainingWatcher):
    async def train_status_websocket_endpoint(websocket: WebSocket) -> None:
        dataset = websocket.query_params.get("dataset")
        if not dataset:
            await websocket.close(code=1008, reason="dataset query param is required")
            return

        job_id = websocket.query_params.get("job_id")

        await websocket.accept()
        uuid, event = training_watcher.subscribe(dataset=dataset, job_id=job_id)
        logging.info("/train/ws websocket connected")

        try:
            initial_value = await training_watcher.get_initial_value(dataset=dataset, job_id=job_id)
            if initial_value is not None:
                await websocket.send_json(dataclasses.asdict(initial_value))

            while True:
                new_value = await event.wait_value()
                event.clear()
                if new_value is not None:
                    await websocket.send_json(dataclasses.asdict(new_value))
        except WebSocketDisconnect:
            pass
        finally:
            training_watcher.unsubscribe(uuid)
            logging.info("/train/ws websocket disconnected")

    return train_status_websocket_endpoint
