from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from valentina.__main__ import build_parser


def test_generate_command_collects_prompts(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("prompt from file\n\nsecond file prompt\n")

    captured: dict[str, object] = {}

    def fake_generate(prompts, config):  # type: ignore[override]
        captured["prompts"] = prompts
        captured["config"] = config
        return []

    monkeypatch.setattr("valentina.__main__.generate_images", fake_generate)

    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "direct prompt",
            "--prompts-file",
            str(prompts_file),
            "--base-model",
            "test/model",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    args.func(args)

    assert captured["prompts"] == [
        "direct prompt",
        "prompt from file",
        "second file prompt",
    ]
    config = captured["config"]
    assert config.base_model == "test/model"  # type: ignore[attr-defined]
    assert Path(config.output_dir) == (tmp_path / "outputs").resolve()  # type: ignore[attr-defined]


def test_generate_command_without_prompts_exits(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--base-model",
            "test/model",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )
    with pytest.raises(SystemExit):
        args.func(args)
