# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

# Ordered by preference within each task family
_TEXT_COLUMN_CANDIDATES = ("text", "sentence", "content", "document", "input", "passage")
_TOKEN_COLUMN_CANDIDATES = ("tokens", "words", "text")
_SEQ2SEQ_INPUT_CANDIDATES = ("input", "source", "question", "text", "sentence")
_SEQ2SEQ_TARGET_CANDIDATES = ("target", "answer", "label", "output", "summary")
_QA_CONTEXT_CANDIDATES = ("context", "passage", "text")
_QA_QUESTION_CANDIDATES = ("question", "query")


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
    for name, ds in zip(splits_to_load, raw if isinstance(raw, list) else [raw]):
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        result[name] = ds

    return result
