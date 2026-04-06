# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict
from peft import TaskType

from worker.training._base import resolve_output_dir, resolve_task
from worker.training._data import (
    build_data_collator,
    ensure_padding_token,
    load_splits,
    resolve_qa_columns,
    resolve_seq2seq_columns,
    resolve_text_column,
    resolve_token_column,
    tokenize_split,
)
from worker.training.algorithms import SUPPORTED_ALGORITHMS, TrainingAlgorithmResult, run_training_algorithm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_context(task_type: str = "text-classification") -> dict[str, object]:
    return {
        "job_id": "job-123",
        "dataset": "stanfordnlp/imdb",
        "revision": "main",
        "model_name": "distilbert-base-uncased",
        "task_type": task_type,
        "train_split": "train",
        "eval_split": "test",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "seed": 42,
        "max_samples": 100,
        "experiment_name": "smoke",
    }


def _make_result() -> TrainingAlgorithmResult:
    return TrainingAlgorithmResult(metrics={"train_loss": 0.5}, artifacts={"trainable_params": 1000})


def _ds(columns: dict[str, list[object]]) -> Dataset:
    return Dataset.from_dict(columns)


# ---------------------------------------------------------------------------
# resolve_task
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_type, expected_peft_task",
    [
        ("text-classification", TaskType.SEQ_CLS),
        ("token-classification", TaskType.TOKEN_CLS),
        ("seq2seq", TaskType.SEQ_2_SEQ_LM),
        ("causal-lm", TaskType.CAUSAL_LM),
        ("question-answering", TaskType.QUESTION_ANS),
    ],
)
def test_resolve_task_returns_correct_peft_type(task_type: str, expected_peft_task: TaskType) -> None:
    _, peft_type = resolve_task(task_type)
    assert peft_type == expected_peft_task


def test_resolve_task_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unsupported task_type"):
        resolve_task("image-classification")


# ---------------------------------------------------------------------------
# resolve_output_dir
# ---------------------------------------------------------------------------


def test_resolve_output_dir_uses_experiment_name(tmp_path: object) -> None:
    with patch("worker.training._base.os.makedirs"):
        path = resolve_output_dir("lora", "my-experiment")
    assert path == "/tmp/training/lora/my-experiment"


def test_resolve_output_dir_falls_back_to_algorithm(tmp_path: object) -> None:
    with patch("worker.training._base.os.makedirs"):
        path = resolve_output_dir("lora", None)
    assert path == "/tmp/training/lora"


def test_resolve_output_dir_uses_run_id_and_configured_root(tmp_path: object) -> None:
    with patch.dict("worker.training._base.os.environ", {"TRAINING_OUTPUT_ROOT": "/vol/checkpoints"}, clear=False):
        with patch("worker.training._base.os.makedirs"):
            path = resolve_output_dir("lora", "my-experiment", run_id="job/1")
    assert path == "/vol/checkpoints/lora/my-experiment/job--1"


# ---------------------------------------------------------------------------
# Column resolvers
# ---------------------------------------------------------------------------


def test_resolve_text_column_finds_text() -> None:
    ds = _ds({"text": ["hello"], "label": [0]})
    assert resolve_text_column(ds) == "text"


def test_resolve_text_column_finds_sentence() -> None:
    ds = _ds({"sentence": ["hello"], "label": [0]})
    assert resolve_text_column(ds) == "sentence"


def test_resolve_text_column_raises_when_missing() -> None:
    ds = _ds({"tokens": [["hello"]], "label": [0]})
    with pytest.raises(ValueError, match="No recognised text column"):
        resolve_text_column(ds)


def test_resolve_token_column_finds_tokens() -> None:
    ds = _ds({"tokens": [["hello", "world"]], "ner_tags": [[0, 1]]})
    assert resolve_token_column(ds) == "tokens"


def test_resolve_token_column_raises_when_missing() -> None:
    ds = _ds({"label": [0]})
    with pytest.raises(ValueError, match="No recognised token column"):
        resolve_token_column(ds)


def test_resolve_seq2seq_columns_finds_input_and_target() -> None:
    ds = _ds({"input": ["translate this"], "target": ["übersetz das"]})
    input_col, target_col = resolve_seq2seq_columns(ds)
    assert input_col == "input"
    assert target_col == "target"


def test_resolve_seq2seq_columns_raises_when_target_missing() -> None:
    ds = _ds({"input": ["translate this"]})
    with pytest.raises(ValueError, match="Could not resolve seq2seq columns"):
        resolve_seq2seq_columns(ds)


def test_resolve_qa_columns_finds_context_and_question() -> None:
    ds = _ds({"context": ["some passage"], "question": ["what?"]})
    ctx_col, q_col = resolve_qa_columns(ds)
    assert ctx_col == "context"
    assert q_col == "question"


def test_resolve_qa_columns_raises_when_missing() -> None:
    ds = _ds({"text": ["some passage"]})
    with pytest.raises(ValueError, match="Could not resolve QA columns"):
        resolve_qa_columns(ds)


def test_tokenize_split_uses_text_target_for_seq2seq() -> None:
    class _FakeTokenizer:
        pad_token_id = 0

        def __call__(self, *args: object, **kwargs: object) -> dict[str, list[list[int]]]:
            if kwargs.get("text_target") is not None:
                assert kwargs["text_target"] == ["target text"]
                return {"input_ids": [[7, 8, 0]]}
            assert args[0] == ["input text"]
            return {"input_ids": [[1, 2, 0]]}

    ds = _ds({"input": ["input text"], "target": ["target text"]})
    tokenized = tokenize_split(ds, _FakeTokenizer(), "seq2seq")
    assert tokenized[0]["labels"] == [7, 8, -100]


def test_tokenize_split_builds_causal_lm_labels() -> None:
    class _FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
            assert texts == ["hello world"]
            return {"input_ids": [[11, 12, 0]]}

    ds = _ds({"text": ["hello world"]})
    tokenized = tokenize_split(ds, _FakeTokenizer(), "causal-lm")
    assert "labels" not in tokenized.column_names


def test_tokenize_split_builds_qa_span_labels() -> None:
    class _FakeQAEncoding(dict):
        def sequence_ids(self, batch_index: int) -> list[int | None]:
            assert batch_index == 0
            return [None, 0, None, 1, 1, None]

    class _FakeTokenizer:
        def __call__(self, questions: list[str], contexts: list[str], **kwargs: object) -> _FakeQAEncoding:
            assert questions == ["what?"]
            assert contexts == ["hello world"]
            assert kwargs.get("return_offsets_mapping") is True
            return _FakeQAEncoding({"offset_mapping": [[(0, 0), (0, 5), (0, 0), (0, 5), (6, 11), (0, 0)]]})

    ds = _ds({"question": ["what?"], "context": ["hello world"], "answers": [{"text": ["world"], "answer_start": [6]}]})
    tokenized = tokenize_split(ds, _FakeTokenizer(), "question-answering")
    assert tokenized[0]["start_positions"] == 4
    assert tokenized[0]["end_positions"] == 4


def test_tokenize_split_raises_for_unsupported_qa_answer_format() -> None:
    class _FakeQAEncoding(dict):
        def sequence_ids(self, batch_index: int) -> list[int | None]:
            assert batch_index == 0
            return [None, 0, None, 1, 1, None]

    class _FakeTokenizer:
        def __call__(self, questions: list[str], contexts: list[str], **kwargs: object) -> _FakeQAEncoding:
            return _FakeQAEncoding({"offset_mapping": [[(0, 0), (0, 5), (0, 0), (0, 5), (6, 11), (0, 0)]]})

    ds = _ds({"question": ["what?"], "context": ["hello world"], "answers": ["world"]})
    with pytest.raises(ValueError, match="Unsupported QA answer format"):
        tokenize_split(ds, _FakeTokenizer(), "question-answering")


def test_tokenize_split_aligns_token_classification_labels() -> None:
    class _FakeEncoding(dict):
        def word_ids(self, batch_index: int) -> list[int | None]:
            assert batch_index == 0
            return [None, 0, 0, 1, None]

    class _FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts: list[list[str]], **kwargs: object) -> _FakeEncoding:
            assert texts == [["hello", "world"]]
            return _FakeEncoding({"input_ids": [[101, 102, 103, 104, 0]]})

    ds = _ds({"tokens": [["hello", "world"]], "ner_tags": [[3, 7]]})
    tokenized = tokenize_split(ds, _FakeTokenizer(), "token-classification")
    assert tokenized[0]["labels"] == [-100, 3, -100, 7, -100]


def test_build_data_collator_only_enables_token_classification() -> None:
    tokenizer = MagicMock()
    assert build_data_collator(tokenizer, "token-classification") is not None
    assert build_data_collator(tokenizer, "causal-lm") is not None
    assert build_data_collator(tokenizer, "text-classification") is None


def test_ensure_padding_token_uses_eos_when_missing() -> None:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "</s>"

    ensure_padding_token(tokenizer)

    assert tokenizer.pad_token == "</s>"


def test_load_splits_handles_single_dataset_return() -> None:
    ds = _ds({"text": ["hello"], "label": [0]})

    with patch("worker.training._data.load_dataset", return_value=DatasetDict({"train": ds})):
        splits = load_splits("dummy", "main", "train", None, None)

    assert list(splits) == ["train"]
    assert splits["train"] is ds


# ---------------------------------------------------------------------------
# QLoRA CPU fallback
# ---------------------------------------------------------------------------


def test_qlora_falls_back_to_lora_on_cpu() -> None:
    from worker.training import qlora

    mock_model = MagicMock()
    mock_model.get_nb_trainable_parameters.return_value = (1000, 10000)

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("worker.training.qlora.resolve_task", return_value=(MagicMock(), MagicMock())),
        patch("worker.training.qlora.get_peft_model", return_value=mock_model),
        patch("worker.training.qlora.prepare_model_for_kbit_training") as mock_kbit,
    ):
        _, _, quantized = qlora._load_qlora_model("distilbert-base-uncased", "text-classification")

    assert quantized is False
    mock_kbit.assert_not_called()


def test_qlora_uses_quantization_on_cuda() -> None:
    from worker.training import qlora

    mock_model = MagicMock()
    mock_model.get_nb_trainable_parameters.return_value = (1000, 10000)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("worker.training.qlora.resolve_task", return_value=(MagicMock(), MagicMock())),
        patch("worker.training.qlora.get_peft_model", return_value=mock_model),
        patch("worker.training.qlora.prepare_model_for_kbit_training", return_value=MagicMock()),
    ):
        _, _, quantized = qlora._load_qlora_model("distilbert-base-uncased", "text-classification")

    assert quantized is True


# ---------------------------------------------------------------------------
# Task restriction enforcement
# ---------------------------------------------------------------------------


def test_linear_probing_rejects_unsupported_task() -> None:
    from worker.training.linear_probing import _load_frozen_model

    with pytest.raises(ValueError, match="Linear probing does not support task_type"):
        _load_frozen_model("distilbert-base-uncased", "causal-lm")


def test_prefix_tuning_rejects_unsupported_task() -> None:
    from worker.training.prefix_tuning import _load_prefix_model

    with pytest.raises(ValueError, match="Prefix tuning does not support task_type"):
        _load_prefix_model("distilbert-base-uncased", "question-answering")


def test_prompt_tuning_rejects_unsupported_task() -> None:
    from worker.training.prompt_tuning import _load_prompt_model

    with pytest.raises(ValueError, match="Prompt tuning does not support task_type"):
        _load_prompt_model("distilbert-base-uncased", "token-classification")


# ---------------------------------------------------------------------------
# Registry dispatch (unchanged)
# ---------------------------------------------------------------------------


def test_run_training_algorithm_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unsupported training algorithm"):
        run_training_algorithm(name="my-algo", context=_get_context())


def test_supported_algorithms_set() -> None:
    assert SUPPORTED_ALGORITHMS == {
        "linear-probing",
        "full-finetune",
        "lora",
        "qlora",
        "prefix-tuning",
        "prompt-tuning",
        "adapter-tuning",
    }


@pytest.mark.parametrize("algorithm_name", sorted(SUPPORTED_ALGORITHMS))
def test_run_training_algorithm_dispatches_to_correct_module(algorithm_name: str) -> None:
    module_map = {
        "linear-probing": "worker.training.linear_probing",
        "full-finetune": "worker.training.full_finetune",
        "lora": "worker.training.lora",
        "qlora": "worker.training.qlora",
        "prefix-tuning": "worker.training.prefix_tuning",
        "prompt-tuning": "worker.training.prompt_tuning",
        "adapter-tuning": "worker.training.adapter_tuning",
    }
    mock_result = _make_result()
    with patch(f"{module_map[algorithm_name]}.run", return_value=mock_result) as mock_run:
        result = run_training_algorithm(name=algorithm_name, context=_get_context())
    mock_run.assert_called_once_with(_get_context())
    assert result == mock_result
