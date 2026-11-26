from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest

from agent_gym import prepare_data
from agent_gym.layout import GymLayout


def _layout(repo: Path, *, data_root: Path | None = None) -> GymLayout:
    return GymLayout(
        repo_root=repo,
        overlay_root=repo / "overlay",
        workspace_root=repo,
        data_root=data_root,
        extra_roots=(),
        framework_root=repo,
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def _populate(repo: Path) -> None:
    for relative in prepare_data.JSONL_PATHS.values():
        _touch(repo / relative)
    _touch(repo / "resources_servers/arc_agi/data/example.jsonl")
    _touch(repo / "benchmarks/aegis_v4_safety/data/source.jsonl")
    _touch(repo / "benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl")
    for prompt in set(prepare_data.PROMPT_CONFIGS.values()):
        _touch(repo / "overlay" / prompt)


class FakeRunner:
    def __init__(self, returns: Sequence[int] = ()) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []
        self.returns = list(returns)

    def __call__(self, command: list[str], *, cwd: Path, env: dict[str, str], check: bool):
        self.calls.append((command, cwd, env))
        code = self.returns.pop(0) if self.returns else 0
        return subprocess.CompletedProcess(command, code)


def test_expand_targets_defaults() -> None:
    assert prepare_data.expand_targets(("a", "b"), []) == ["a", "b"]
    assert prepare_data.expand_targets(("a", "b"), ["b"]) == ["b"]
    assert prepare_data.expand_targets(("a", "b"), ["all"]) == ["a", "b"]
    assert prepare_data.expand_targets(("a", "b"), ["b", "all", "b"]) == ["b", "a"]


def test_usage_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare_data.main([]) == 2
    assert "Usage: agent-gym prepare-data" in capsys.readouterr().err

    assert prepare_data.main(["--help"]) == 0
    assert "Usage: agent-gym prepare-data" in capsys.readouterr().out

    assert prepare_data.main(["prepare", "--help", "arc-agi"]) == 0
    assert "Commands:" in capsys.readouterr().out


def test_only_dry_run_reports_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare_data.main(["--dry-run"], environ={}) == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.err
    assert "unknown command:" in captured.err


def test_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare_data.main(["bogus"], environ={}) == 1
    assert "unknown command: bogus" in capsys.readouterr().err


def test_all_rejects_names(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare_data.main(["all", "gsm8k"], environ={}) == 1
    assert "all does not accept benchmark names" in capsys.readouterr().err


def test_dry_run_all_prints_full_command_sequence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    _populate(repo)
    runner = FakeRunner()

    assert prepare_data.main(["--dry-run", "all"], layout=_layout(repo), runner=runner, environ={}) == 0

    assert runner.calls == []
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("+ ")]
    assert lines == [
        "+ agent-gym eval prepare --benchmark aegis_v4_safety",
        "+ agent-gym eval prepare --benchmark arc-agi",
        "+ agent-gym eval prepare --benchmark arc-agi-v2",
        "+ agent-gym eval prepare --benchmark arc-agi-v3",
        "+ agent-gym eval prepare --benchmark gaia",
        "+ agent-gym eval prepare --benchmark gsm8k",
        f"+ agent-gym dataset collate --config benchmarks/aegis_v4_safety/config.yaml "
        f"--output-dir {repo / '.tmp/data-prep/aegis_v4_safety'} --mode train_preparation",
        f"+ agent-gym dataset collate --config benchmarks/arc-agi/config.yaml "
        f"--output-dir {repo / '.tmp/data-prep/arc-agi'} --mode train_preparation",
        f"+ agent-gym dataset collate --config benchmarks/arc-agi-v2/config.yaml "
        f"--output-dir {repo / '.tmp/data-prep/arc-agi-v2'} --mode train_preparation",
        f"+ agent-gym dataset collate --config benchmarks/gsm8k/config.yaml "
        f"--output-dir {repo / '.tmp/data-prep/gsm8k'} --mode train_preparation",
        f"+ agent-gym dataset render --input {repo / 'benchmarks/arc-agi/data/arc_agi_benchmark.jsonl'} "
        f"--prompt-config {repo / 'overlay/benchmarks/prompts/generic/arc.yaml'} "
        f"--output {repo / 'benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl'}",
        f"+ agent-gym dataset render --input {repo / 'benchmarks/arc-agi-v2/data/arc_agi_v2_benchmark.jsonl'} "
        f"--prompt-config {repo / 'overlay/benchmarks/prompts/generic/arc.yaml'} "
        f"--output {repo / 'benchmarks/arc-agi-v2/data/arc_agi_v2_benchmark_materialized.jsonl'}",
        f"+ agent-gym dataset render --input {repo / 'benchmarks/gsm8k/data/gsm8k_benchmark.jsonl'} "
        f"--prompt-config {repo / 'overlay/benchmarks/prompts/generic/math.yaml'} "
        f"--output {repo / 'benchmarks/gsm8k/data/gsm8k_benchmark_materialized.jsonl'}",
        f"+ {sys.executable} scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py "
        f"{repo / 'benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl'} "
        f"--train {repo / 'benchmarks/arc-agi/data/rsi/arc_agi_train.jsonl'} "
        f"--holdout {repo / 'benchmarks/arc-agi/data/rsi/arc_agi_holdout.jsonl'} --id-field task_id",
        f"+ {sys.executable} scripts/run_component.py benchmarks/rsi_dataset_tools/split_jsonl.py "
        f"{repo / 'benchmarks/gaia/data/gaia_text_validation.jsonl'} "
        f"--train {repo / 'benchmarks/gaia/data/rsi/gaia_train.jsonl'} "
        f"--holdout {repo / 'benchmarks/gaia/data/rsi/gaia_holdout.jsonl'} --id-field task_id",
    ]


def test_prepare_runs_one_process_per_benchmark_with_environment(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _populate(repo)
    runner = FakeRunner()

    assert prepare_data.main(["prepare", "arc-agi", "arc-agi-v2"], layout=_layout(repo), runner=runner, environ={}) == 0

    commands = [call[0] for call in runner.calls]
    assert commands == [
        ["agent-gym", "eval", "prepare", "--benchmark", "arc-agi"],
        ["agent-gym", "eval", "prepare", "--benchmark", "arc-agi-v2"],
    ]
    _, cwd, env = runner.calls[0]
    assert cwd == repo
    assert env["AGENT_GYM_REPO"] == str(repo)
    assert env["AGENT_GYM_OVERLAY"] == str(repo / "overlay")
    assert env["PYTHONPATH"] == f"{repo / 'overlay'}{':'}{repo}"


def test_prepare_uses_configured_binaries_and_existing_pythonpath(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runner = FakeRunner()

    environ = {"GYM_BIN": "/opt/gym/bin/gym", "PYTHON_BIN": "/opt/python", "PYTHONPATH": "/existing"}
    assert prepare_data.main(["prepare", "gaia"], layout=_layout(repo), runner=runner, environ=environ) == 0

    assert runner.calls[0][0] == ["/opt/gym/bin/gym", "eval", "prepare", "--benchmark", "gaia"]
    assert runner.calls[0][2]["PYTHONPATH"] == f"{repo / 'overlay'}:{repo}:/existing"


def test_prepare_seeds_arc_example_before_preparing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runner = FakeRunner()

    assert prepare_data.main(["prepare", "arc-agi"], layout=_layout(repo), runner=runner, environ={}) == 0

    assert [call[0] for call in runner.calls] == [
        [sys.executable, "scripts/run_component.py", "resources_servers/arc_agi/prepare_example.py"],
        ["agent-gym", "eval", "prepare", "--benchmark", "arc-agi"],
    ]


def test_prepare_copies_aegis_source_from_data_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    wheel = tmp_path / "wheel"
    relative = "benchmarks/aegis_v4_safety/data/source.jsonl"
    source = _touch(wheel / relative)
    source.write_text("from-wheel\n", encoding="utf-8")
    runner = FakeRunner()

    layout = _layout(repo, data_root=wheel)
    assert prepare_data.main(["prepare", "aegis_v4_safety"], layout=layout, runner=runner, environ={}) == 0

    destination = repo / relative
    assert destination.read_text(encoding="utf-8") == "from-wheel\n"
    assert f"Seeded {relative} from {source}" in capsys.readouterr().out
    assert [call[0] for call in runner.calls] == [
        ["agent-gym", "eval", "prepare", "--benchmark", "aegis_v4_safety"]
    ]


def test_prepare_fails_when_aegis_source_cannot_be_resolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    assert prepare_data.main(["prepare", "aegis_v4_safety"], layout=_layout(repo), environ={}) == 1
    assert "cannot resolve benchmarks/aegis_v4_safety/data/source.jsonl" in capsys.readouterr().err


def test_prepare_rejects_unknown_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    assert prepare_data.main(["prepare", "nope"], layout=_layout(repo), environ={}) == 1
    assert "unknown benchmark: nope" in capsys.readouterr().err


def test_prepare_stops_at_first_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runner = FakeRunner(returns=[3])

    assert prepare_data.main(["prepare", "gaia", "gsm8k"], layout=_layout(repo), runner=runner, environ={}) == 3
    assert len(runner.calls) == 1


def test_collate_uses_env_output_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _touch(repo / prepare_data.JSONL_PATHS["gsm8k"])
    runner = FakeRunner()

    assert (
        prepare_data.main(
            ["collate", "gsm8k"],
            layout=_layout(repo),
            runner=runner,
            environ={"AGENT_GYM_PREPARE_OUTPUT": "custom"},
        )
        == 0
    )
    assert runner.calls[0][0] == [
        "agent-gym",
        "dataset",
        "collate",
        "--config",
        "benchmarks/gsm8k/config.yaml",
        "--output-dir",
        str(repo / "custom" / "gsm8k"),
        "--mode",
        "train_preparation",
    ]


def test_collate_rejects_unsupported_and_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"

    assert prepare_data.main(["collate", "gaia"], layout=_layout(repo), environ={}) == 1
    assert "collate is not supported for gaia" in capsys.readouterr().err

    assert prepare_data.main(["collate", "gsm8k"], layout=_layout(repo), environ={}) == 1
    assert (
        "missing benchmarks/gsm8k/data/gsm8k_benchmark.jsonl; "
        "run: bash scripts/prepare_data.sh prepare gsm8k" in capsys.readouterr().err
    )


def test_render_rejects_unsupported_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    assert prepare_data.main(["render", "gaia"], layout=_layout(repo), environ={}) == 1
    assert "render is not defined for gaia; supported: arc-agi arc-agi-v2 gsm8k" in capsys.readouterr().err


def test_rsi_command_and_missing_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    source = _touch(repo / prepare_data.RSI_SOURCES["gaia"])
    runner = FakeRunner()

    assert prepare_data.main(["rsi", "gaia"], layout=_layout(repo), runner=runner, environ={}) == 0
    assert runner.calls[0][0] == [
        sys.executable,
        "scripts/run_component.py",
        "benchmarks/rsi_dataset_tools/split_jsonl.py",
        str(source),
        "--train",
        str(repo / "benchmarks/gaia/data/rsi/gaia_train.jsonl"),
        "--holdout",
        str(repo / "benchmarks/gaia/data/rsi/gaia_holdout.jsonl"),
        "--id-field",
        "task_id",
    ]

    assert prepare_data.main(["rsi", "arc-agi"], layout=_layout(repo), environ={}) == 1
    assert "missing benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl" in capsys.readouterr().err


def test_rsi_rejects_unsupported_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    assert prepare_data.main(["rsi", "gsm8k"], layout=_layout(repo), environ={}) == 1
    assert "RSI split is not defined for gsm8k; supported: arc-agi gaia" in capsys.readouterr().err
