# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from collections.abc import Callable, Mapping
from functools import lru_cache
import importlib
import inspect
import time
from typing import Any, Optional, TypedDict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainingExecutionContext(TypedDict):
    job_id: str
    dataset: str
    revision: str
    model_name: str
    task_type: str
    train_split: str
    eval_split: Optional[str]
    epochs: int
    batch_size: int
    learning_rate: float
    seed: Optional[int]
    max_samples: Optional[int]
    experiment_name: Optional[str]
    local_dataset_path: Optional[str]
    local_dataset_format: Optional[str]
    cancellation_checker: Optional[Callable[[], bool]]
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]]
    # Optional resource allocation overrides
    use_gpu: Optional[bool]
    gpu_count: Optional[int]
    gpu_type: Optional[str]
    cpu_cores: Optional[int]
    memory_gb: Optional[int]


class TrainingAlgorithmResult(TypedDict):
    metrics: Mapping[str, float]
    artifacts: Mapping[str, Any]


class TrainingCancelledError(RuntimeError):
    pass


class CancellationCallback(TrainerCallback):
    def __init__(self, cancellation_checker: Callable[[], bool]):
        self._cancellation_checker = cancellation_checker
        self.cancelled = False

    def _check(self, control: TrainerControl) -> TrainerControl:
        if self._cancellation_checker():
            self.cancelled = True
            control.should_training_stop = True
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        return self._check(control)


class ProgressCallback(TrainerCallback):
    def __init__(self, progress_callback: Callable[[Mapping[str, Any]], None]):
        self._progress_callback = progress_callback
        self._started_at = time.monotonic()

    def _emit(
        self,
        *,
        event: str,
        state: TrainerState,
        control: TrainerControl,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> TrainerControl:
        max_steps = int(state.max_steps or 0)
        global_step = int(state.global_step or 0)
        progress_pct = float(global_step / max_steps * 100.0) if max_steps > 0 else 0.0

        eta_seconds: Optional[float] = None
        if max_steps > 0 and global_step > 0:
            elapsed = max(0.001, time.monotonic() - self._started_at)
            steps_per_second = global_step / elapsed
            if steps_per_second > 0:
                remaining_steps = max(0, max_steps - global_step)
                eta_seconds = float(remaining_steps / steps_per_second)

        payload: dict[str, Any] = {
            "event": event,
            "global_step": global_step,
            "max_steps": max_steps,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "progress_pct": progress_pct,
            "eta_seconds": eta_seconds,
        }
        if extra:
            payload.update(dict(extra))

        self._progress_callback(payload)
        return control

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        self._started_at = time.monotonic()
        return self._emit(event="train_begin", state=state, control=control)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # Emit every 10 steps and on the final step to keep updates informative but lightweight.
        max_steps = int(state.max_steps or 0)
        global_step = int(state.global_step or 0)
        if global_step % 10 != 0 and (max_steps <= 0 or global_step < max_steps):
            return control
        return self._emit(event="train_step", state=state, control=control)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        return self._emit(
            event="train_end",
            state=state,
            control=control,
            extra={"progress_pct": 100.0, "eta_seconds": 0.0},
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        return self._emit(event="epoch_end", state=state, control=control)


def build_cancellation_callback(context: TrainingExecutionContext) -> Optional[CancellationCallback]:
    cancellation_checker = context.get("cancellation_checker")
    if cancellation_checker is None:
        return None
    return CancellationCallback(cancellation_checker=cancellation_checker)


def build_progress_callback(context: TrainingExecutionContext) -> Optional[ProgressCallback]:
    progress_callback = context.get("progress_callback")
    if progress_callback is None:
        return None
    return ProgressCallback(progress_callback=progress_callback)


@lru_cache(maxsize=1)
def _validate_accelerate_runtime() -> None:
    # Transformers Trainer expects accelerate.Accelerator.unwrap_model to support
    # keep_torch_compile; older accelerate versions raise a runtime TypeError.
    accelerator_module = importlib.import_module("accelerate")
    accelerator_class = getattr(accelerator_module, "Accelerator")
    unwrap_signature = inspect.signature(accelerator_class.unwrap_model)
    if "keep_torch_compile" not in unwrap_signature.parameters:
        raise RuntimeError(
            "Incompatible accelerate runtime detected. Upgrade accelerate to >=0.31.0 to run training jobs."
        )


TrainingAlgorithm = Callable[[TrainingExecutionContext], TrainingAlgorithmResult]


def _build_registry() -> dict[str, TrainingAlgorithm]:
    # Deferred imports so algorithm modules can import from this module without circular deps
    from worker.training import (
        adapter_tuning,
        full_finetune,
        linear_probing,
        lora,
        prefix_tuning,
        prompt_tuning,
        qlora,
    )

    return {
        "linear-probing": linear_probing.run,
        "full-finetune": full_finetune.run,
        "lora": lora.run,
        "qlora": qlora.run,
        "prefix-tuning": prefix_tuning.run,
        "prompt-tuning": prompt_tuning.run,
        "adapter-tuning": adapter_tuning.run,
    }


def run_training_algorithm(name: str, context: TrainingExecutionContext) -> TrainingAlgorithmResult:
    _validate_accelerate_runtime()

    registry = _build_registry()
    algorithm = registry.get(name)
    if algorithm is None:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported training algorithm '{name}'. Supported algorithms: {supported}")
    return algorithm(context)


# Expose supported names without importing algorithm modules
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    ["linear-probing", "full-finetune", "lora", "qlora", "prefix-tuning", "prompt-tuning", "adapter-tuning"]
)
