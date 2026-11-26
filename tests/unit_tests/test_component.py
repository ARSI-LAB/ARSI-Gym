from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_gym import component


def test_component_script_accepts_known_roots() -> None:
    assert str(component._component_script("benchmarks/osworld/prepare.py")) == "benchmarks/osworld/prepare.py"
    assert str(component._component_script("resources_servers/arc_agi/prepare_example.py")) == (
        "resources_servers/arc_agi/prepare_example.py"
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/abs/benchmarks/prepare.py",
        "../benchmarks/prepare.py",
        "benchmarks/../../etc/passwd",
        "scripts/run_component.py",
        "other/prepare.py",
    ],
)
def test_component_script_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError):
        component._component_script(value)


def test_command_dispatches_by_suffix(tmp_path: Path) -> None:
    py = tmp_path / "prepare.py"
    sh = tmp_path / "prepare.sh"
    other = tmp_path / "prepare"

    assert component._command(py, ["--x"]) == [sys.executable, str(py), "--x"]
    assert component._command(sh, ["--x"]) == ["bash", str(sh), "--x"]
    assert component._command(other, ["--x"]) == [str(other), "--x"]


def test_main_without_arguments_prints_usage_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert component.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage: agent-gym component run" in captured.err


def test_main_help_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert component.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage: agent-gym component run" in captured.out
    assert captured.err == ""


def test_main_rejects_bad_script_path(capsys: pytest.CaptureFixture[str]) -> None:
    assert component.main(["scripts/run_component.py"]) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_main_runs_staged_script_with_workspace_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    script = workspace / "benchmarks" / "demo" / "prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('hi')\n", encoding="utf-8")

    layout = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(component, "stage_prepare_components", lambda active: workspace)
    monkeypatch.setattr(component, "cleanup_staged_components", lambda active: calls.setdefault("cleaned", active))
    monkeypatch.setenv("PYTHONPATH", "/existing")

    def fake_run(command: list[str], *, cwd: Path, env: dict[str, str], check: bool):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env"] = env
        calls["check"] = check
        return subprocess.CompletedProcess(command, 7)

    assert component.main(["benchmarks/demo/prepare.py", "--flag"], layout=layout, runner=fake_run) == 7
    assert calls["command"] == [sys.executable, str(script), "--flag"]
    assert calls["cwd"] == workspace
    assert calls["check"] is False
    assert calls["env"]["PYTHONPATH"] == f"{workspace}{os.pathsep}/existing"
    assert calls["cleaned"] is layout


def test_main_reports_missing_staged_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cleaned: list[object] = []
    layout = object()

    monkeypatch.setattr(component, "stage_prepare_components", lambda active: workspace)
    monkeypatch.setattr(component, "cleanup_staged_components", lambda active: cleaned.append(active))

    assert component.main(["benchmarks/demo/missing.py"], layout=layout, runner=lambda *a, **k: None) == 2
    captured = capsys.readouterr()
    assert "staged component script not found" in captured.err
    assert cleaned == [layout]


def test_main_prepares_runtime_when_layout_omitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    script = workspace / "benchmarks" / "demo" / "prepare.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    prepared: dict[str, object] = {}
    layout = object()

    def fake_prepare_runtime(**kwargs):
        prepared.update(kwargs)
        return layout

    monkeypatch.setattr(component, "prepare_runtime", fake_prepare_runtime)
    monkeypatch.setattr(component, "stage_prepare_components", lambda active: workspace)
    monkeypatch.setattr(component, "cleanup_staged_components", lambda active: None)

    assert component.main(["benchmarks/demo/prepare.py"], runner=lambda *a, **k: subprocess.CompletedProcess([], 0)) == 0
    assert prepared == {
        "require_overlay": True,
        "verify_trees": True,
        "chdir": False,
        "import_framework": False,
    }
