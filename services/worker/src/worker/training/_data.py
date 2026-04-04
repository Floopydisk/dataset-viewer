# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForTokenClassification

# Ordered by preference within each task family
_TEXT_COLUMN_CANDIDATES = ("text", "sentence", "content", "document", "input", "passage")
_TOKEN_COLUMN_CANDIDATES = ("tokens", "words", "text")
_TOKEN_LABEL_COLUMN_CANDIDATES = ("labels", "label", "ner_tags", "tags", "pos_tags")
_SEQ2SEQ_INPUT_CANDIDATES = ("input", "source", "question", "text", "sentence")
_SEQ2SEQ_TARGET_CANDIDATES = ("target", "answer", "label", "output", "summary")
_QA_CONTEXT_CANDIDATES = ("context", "passage", "text")
_QA_QUESTION_CANDIDATES = ("question", "query")
_QA_ANSWER_CANDIDATES = ("answers", "answer", "label")


def _first_match(candidates: tuple[str, ...], column_names: list[str]) -> Optional[str]:
    for c in candidates:
        if c in column_names:
            return c
    return None


def resolve_text_column(dataset: Dataset) -> str:
    """Return the text column for single-sequence tasks (text-classification, causal-lm).

    Raises:
        ValueError: if no recognised column is found.
    """
    col = _first_match(_TEXT_COLUMN_CANDIDATES, dataset.column_names)
    if col is None:
        raise ValueError(
            f"No recognised text column found. Available: {dataset.column_names}. "
            f"Expected one of: {list(_TEXT_COLUMN_CANDIDATES)}"
        )
    return col


def resolve_token_column(dataset: Dataset) -> str:
    """Return the token-sequence column for token-classification tasks.

    Raises:
        ValueError: if no recognised column is found.
    """
    col = _first_match(_TOKEN_COLUMN_CANDIDATES, dataset.column_names)
    if col is None:
        raise ValueError(
            f"No recognised token column found. Available: {dataset.column_names}. "
            f"Expected one of: {list(_TOKEN_COLUMN_CANDIDATES)}"
        )
    return col


def resolve_token_label_column(dataset: Dataset) -> str:
    """Return the token-level label column for token-classification tasks.

    Raises:
        ValueError: if no recognised label column is found.
    """
    col = _first_match(_TOKEN_LABEL_COLUMN_CANDIDATES, dataset.column_names)
    if col is None:
        raise ValueError(
            f"No recognised token-label column found. Available: {dataset.column_names}. "
            f"Expected one of: {list(_TOKEN_LABEL_COLUMN_CANDIDATES)}"
        )
    return col


def resolve_seq2seq_columns(dataset: Dataset) -> tuple[str, str]:
    """Return (input_column, target_column) for seq2seq tasks.

    Raises:
        ValueError: if either column cannot be resolved.
    """
    input_col = _first_match(_SEQ2SEQ_INPUT_CANDIDATES, dataset.column_names)
    target_col = _first_match(_SEQ2SEQ_TARGET_CANDIDATES, dataset.column_names)
    missing = []
    if input_col is None:
        missing.append(f"input (tried {list(_SEQ2SEQ_INPUT_CANDIDATES)})")
    if target_col is None:
        missing.append(f"target (tried {list(_SEQ2SEQ_TARGET_CANDIDATES)})")
    if missing:
        raise ValueError(
            f"Could not resolve seq2seq columns. Missing: {missing}. Available: {dataset.column_names}"
        )
    return input_col, target_col  # type: ignore[return-value]


def resolve_qa_columns(dataset: Dataset) -> tuple[str, str]:
    """Return (context_column, question_column) for question-answering tasks.

    Raises:
        ValueError: if either column cannot be resolved.
    """
    ctx_col = _first_match(_QA_CONTEXT_CANDIDATES, dataset.column_names)
    q_col = _first_match(_QA_QUESTION_CANDIDATES, dataset.column_names)
    missing = []
    if ctx_col is None:
        missing.append(f"context (tried {list(_QA_CONTEXT_CANDIDATES)})")
    if q_col is None:
        missing.append(f"question (tried {list(_QA_QUESTION_CANDIDATES)})")
    if missing:
        raise ValueError(
            f"Could not resolve QA columns. Missing: {missing}. Available: {dataset.column_names}"
        )
    return ctx_col, q_col  # type: ignore[return-value]


def resolve_qa_answer_column(dataset: Dataset) -> str:
    """Return the answer-span column for extractive question-answering tasks."""
    col = _first_match(_QA_ANSWER_CANDIDATES, dataset.column_names)
    if col is None:
        raise ValueError(
            f"No recognised QA answer column found. Available: {dataset.column_names}. "
            f"Expected one of: {list(_QA_ANSWER_CANDIDATES)}"
        )
    return col


def _mask_padding_tokens(input_ids: list[list[int]], pad_token_id: Optional[int]) -> list[list[int]]:
    if pad_token_id is None:
        return input_ids
    return [[token_id if token_id != pad_token_id else -100 for token_id in sequence] for sequence in input_ids]


def _resolve_mask_token_id(tokenizer: object) -> Optional[int]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id
    return getattr(tokenizer, "eos_token_id", None)


def ensure_padding_token(tokenizer: object) -> None:
    """Ensure the tokenizer can pad; fall back to EOS when needed."""
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        raise ValueError(
            "Tokenizer has no pad_token_id and no eos_token. "
            "Please provide a tokenizer with a valid padding token."
        )
    tokenizer.pad_token = eos_token  # type: ignore[attr-defined]


def _extract_qa_answer(answer_value: object) -> tuple[Optional[str], Optional[int]]:
    if isinstance(answer_value, dict):
        texts = answer_value.get("text") or answer_value.get("texts")
        starts = answer_value.get("answer_start") or answer_value.get("answer_starts")
        if isinstance(texts, list):
            answer_text = texts[0] if texts else None
        elif isinstance(texts, str):
            answer_text = texts
        else:
            answer_text = None

        if isinstance(starts, list):
            answer_start = starts[0] if starts else None
        elif isinstance(starts, int):
            answer_start = starts
        else:
            answer_start = None

        if answer_text is not None and answer_start is not None:
            return answer_text, answer_start
        if answer_text is None and answer_start is None:
            return None, None
    raise ValueError(
        "Unsupported QA answer format. Expected a dict with 'text'/'texts' and "
        "'answer_start'/'answer_starts' (scalar or list)."
    )


def tokenize_split(dataset: Dataset, tokenizer: object, task_type: str) -> Dataset:
    """Tokenize a split according to the task type.

    The returned dataset includes task-appropriate labels for seq2seq, causal LM,
    and token-classification tasks.
    """

    if task_type == "token-classification":
        text_col = resolve_token_column(dataset)
        label_col = resolve_token_label_column(dataset)

        def _tok(batch: dict[str, object]) -> dict[str, object]:
            tokenized = tokenizer(  # type: ignore[operator]
                batch[text_col],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
            )
            labels: list[list[int]] = []
            for batch_index, word_labels in enumerate(batch[label_col]):  # type: ignore[index]
                word_ids = tokenized.word_ids(batch_index=batch_index)
                previous_word_idx = None
                aligned_labels: list[int] = []
                for word_idx in word_ids:
                    if word_idx is None:
                        aligned_labels.append(-100)
                    elif word_idx != previous_word_idx:
                        aligned_labels.append(word_labels[word_idx])
                    else:
                        aligned_labels.append(-100)
                    previous_word_idx = word_idx
                labels.append(aligned_labels)
            tokenized["labels"] = labels
            return tokenized

    elif task_type == "seq2seq":
        input_col, target_col = resolve_seq2seq_columns(dataset)

        def _tok(batch: dict[str, object]) -> dict[str, object]:
            model_inputs = tokenizer(  # type: ignore[operator]
                batch[input_col], truncation=True, padding="max_length", max_length=128
            )
            labels = tokenizer(  # type: ignore[operator]
                text_target=batch[target_col], truncation=True, padding="max_length", max_length=128
            )["input_ids"]
            model_inputs["labels"] = _mask_padding_tokens(labels, _resolve_mask_token_id(tokenizer))
            return model_inputs

    elif task_type == "question-answering":
        ctx_col, q_col = resolve_qa_columns(dataset)
        answer_col = resolve_qa_answer_column(dataset)

        def _tok(batch: dict[str, object]) -> dict[str, object]:
            tokenized = tokenizer(  # type: ignore[operator]
                batch[q_col],
                batch[ctx_col],
                truncation="only_second",
                padding="max_length",
                max_length=384,
                return_offsets_mapping=True,
            )

            offset_mapping = tokenized.pop("offset_mapping")
            start_positions: list[int] = []
            end_positions: list[int] = []

            for batch_index, offsets in enumerate(offset_mapping):
                sequence_ids = tokenized.sequence_ids(batch_index)
                answer_text, answer_start = _extract_qa_answer(batch[answer_col][batch_index])  # type: ignore[index]
                if answer_text is None or answer_start is None:
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                answer_end = answer_start + len(answer_text)
                context_start = 0
                while sequence_ids[context_start] != 1:
                    context_start += 1
                context_end = len(sequence_ids) - 1
                while sequence_ids[context_end] != 1:
                    context_end -= 1

                if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                token_start_index = context_start
                while token_start_index <= context_end and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                token_end_index = token_start_index - 1
                while token_end_index <= context_end and offsets[token_end_index][1] < answer_end:
                    token_end_index += 1

                start_positions.append(token_start_index - 1)
                end_positions.append(token_end_index)

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions
            return tokenized

    else:
        text_col = resolve_text_column(dataset)

        def _tok(batch: dict[str, object]) -> dict[str, object]:
            if task_type == "causal-lm":
                return tokenizer(batch[text_col], truncation=True, max_length=128)  # type: ignore[operator]
            return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=128)  # type: ignore[operator]

    return dataset.map(_tok, batched=True)


def build_data_collator(tokenizer: object, task_type: str) -> Optional[object]:
    if task_type == "token-classification":
        return DataCollatorForTokenClassification(tokenizer=tokenizer)  # type: ignore[arg-type]
    if task_type == "causal-lm":
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # type: ignore[arg-type]
    return None


def load_splits(
    dataset: str,
    revision: str,
    train_split: str,
    eval_split: Optional[str],
    max_samples: Optional[int],
) -> DatasetDict:
    """Load train (and optionally eval) splits, applying max_samples if set."""
    splits_to_load = [train_split]
    if eval_split:
        splits_to_load.append(eval_split)

    logging.info(f"Loading dataset {dataset!r} splits={splits_to_load} revision={revision!r}")
    raw = load_dataset(dataset, revision=revision, split=splits_to_load)  # type: ignore[call-overload]

    result: DatasetDict = DatasetDict()
    if isinstance(raw, Dataset):
        if len(splits_to_load) != 1:
            raise ValueError("load_dataset returned a single Dataset for multiple requested splits")
        raw_splits = [raw]
    elif isinstance(raw, DatasetDict):
        raw_splits = [raw[name] for name in splits_to_load]
    else:
        raw_splits = raw
    for name, ds in zip(splits_to_load, raw_splits):
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        result[name] = ds

    return result
