# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from pathlib import Path
from unittest.mock import MagicMock

from worker.training._trainer import train_with_resume


def test_train_with_resume_starts_fresh_when_no_checkpoint(tmp_path: Path) -> None:
    trainer = MagicMock()
    trainer.train.return_value = {"status": "fresh"}

    output, resume = train_with_resume(trainer=trainer, output_dir=str(tmp_path))

    trainer.train.assert_called_once_with()
    assert output == {"status": "fresh"}
    assert resume == {
        "resumed_from_checkpoint": False,
        "resume_checkpoint_path": None,
    }


def test_train_with_resume_uses_latest_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-2").mkdir()

    trainer = MagicMock()
    trainer.train.return_value = {"status": "resumed"}

    output, resume = train_with_resume(trainer=trainer, output_dir=str(tmp_path))

    trainer.train.assert_called_once_with(resume_from_checkpoint=str(tmp_path / "checkpoint-10"))
    assert output == {"status": "resumed"}
    assert resume == {
        "resumed_from_checkpoint": True,
        "resume_checkpoint_path": str(tmp_path / "checkpoint-10"),
    }


def test_train_with_resume_falls_back_after_resume_error(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-42").mkdir()

    trainer = MagicMock()
    trainer.train.side_effect = [RuntimeError("resume failed"), {"status": "fresh-after-error"}]

    output, resume = train_with_resume(trainer=trainer, output_dir=str(tmp_path))

    assert trainer.train.call_count == 2
    assert trainer.train.call_args_list[0].kwargs == {"resume_from_checkpoint": str(tmp_path / "checkpoint-42")}
    assert trainer.train.call_args_list[1].kwargs == {}
    assert output == {"status": "fresh-after-error"}
    assert resume == {
        "resumed_from_checkpoint": False,
        "resume_checkpoint_path": str(tmp_path / "checkpoint-42"),
    }
