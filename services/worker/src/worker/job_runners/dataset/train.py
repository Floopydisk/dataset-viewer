# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import time
from typing import Any, Mapping, Optional

from libcommon.queue.jobs import Queue
from libcommon.train import normalize_training_params

from worker.config import AppConfig
from worker.dtos import CompleteJobResult, JobResult
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.training.algorithms import TrainingCancelledError, TrainingExecutionContext, run_training_algorithm


class DatasetTrainJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-train"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    def _log_training_context(self, message: str) -> None:
        logging.info(
            f"{message} job_id={self.job_info['job_id']} dataset={self.dataset} "
            f"revision={self.dataset_git_revision}"
        )

    def _run_training(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: Optional[int],
    ) -> dict[str, float]:
        # This adapter isolates training execution. It can later call a real trainer backend.
        for epoch in range(1, epochs + 1):
            self._log_training_context(f"Epoch {epoch}/{epochs} in progress...")

        return {
            "accuracy": 0.85 + (epochs * 0.01),
            "loss": 0.1 / epochs,
        }

    def compute(self) -> JobResult:
        self._log_training_context(f"compute {self.get_job_type()}")

        raw_params = self.job_info["params"].get("params_dict")
        if isinstance(raw_params, Mapping):
            params_dict: Mapping[str, Any] = raw_params
        else:
            params_dict = {}

        training_params = normalize_training_params(params_dict, strict=False)
        model_name = training_params["model_name"]
        epochs = training_params["epochs"]
        batch_size = training_params["batch_size"]
        learning_rate = training_params["learning_rate"]
        seed = training_params["seed"]
        task_type = training_params["task_type"]
        training_algorithm = training_params["training_algorithm"]
        train_split = training_params["train_split"]
        eval_split = training_params["eval_split"]
        max_samples = training_params["max_samples"]
        experiment_name = training_params["experiment_name"]
        local_dataset_path = training_params["local_dataset_path"]
        local_dataset_format = training_params["local_dataset_format"]

        logging.info(
            f"Training parameters job_id={self.job_info['job_id']} model_name={model_name} epochs={epochs} "
            f"batch_size={batch_size} learning_rate={learning_rate} task_type={task_type} "
            f"algorithm={training_algorithm}"
        )

        context: TrainingExecutionContext = {
            "job_id": self.job_info["job_id"],
            "dataset": self.dataset,
            "revision": self.dataset_git_revision,
            "model_name": model_name,
            "task_type": task_type,
            "train_split": train_split,
            "eval_split": eval_split,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_samples": max_samples,
            "experiment_name": experiment_name,
            "local_dataset_path": local_dataset_path,
            "local_dataset_format": local_dataset_format,
            "cancellation_checker": lambda: Queue().is_job_cancellation_requested(self.job_info["job_id"]),
        }

        if training_algorithm:
            try:
                algorithm_result = run_training_algorithm(name=training_algorithm, context=context)
                metrics = dict(algorithm_result["metrics"])
                artifacts = dict(algorithm_result["artifacts"])
                result_status = "success"
                result_message = f"Training job completed for {self.dataset}"
            except TrainingCancelledError:
                metrics = {}
                artifacts = {}
                result_status = "cancelled"
                result_message = f"Training job was cancelled for {self.dataset}"
        else:
            metrics = self._run_training(
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
            )
            artifacts = {}
            result_status = "success"
            result_message = f"Training job completed for {self.dataset}"

        return CompleteJobResult(
            content={
                "status": result_status,
                "message": result_message,
                "model_name": model_name,
                "task_type": task_type,
                "training_algorithm": training_algorithm,
                "train_split": train_split,
                "eval_split": eval_split,
                "max_samples": max_samples,
                "experiment_name": experiment_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metrics": metrics,
                "artifacts": artifacts,
            }
        )
