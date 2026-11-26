from __future__ import annotations

from types import SimpleNamespace

import nemo_gym.cli.legacy as legacy_module
import nemo_gym.cli.main as upstream_main_module
import pytest

import agent_gym.cli as cli_module
from agent_gym import doctor, launcher


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"prepare": [], "cleanup": [], "stage": [], "activate": []}
    monkeypatch.setattr(launcher, "GymLayout", SimpleNamespace(discover=lambda: "layout"))
    monkeypatch.setattr(launcher, "_prepare_runtime", lambda **kwargs: calls["prepare"].append(kwargs) or "layout")
    monkeypatch.setattr(launcher, "cleanup_staged_components", lambda layout: calls["cleanup"].append(layout))
    monkeypatch.setattr(launcher, "stage_prepare_components", lambda layout: calls["stage"].append(layout) or layout)
    monkeypatch.setattr(launcher, "activate_staged_components", lambda layout: calls["activate"].append(layout))
    monkeypatch.setattr(launcher, "_needs_prepare_staging", lambda name, argv: False)
    return calls


def _argv(monkeypatch: pytest.MonkeyPatch, *values: str) -> None:
    monkeypatch.setattr(launcher.sys, "argv", list(values))


def test_project_command_intercepted_before_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]
) -> None:
    _argv(monkeypatch, "agent-gym", "service", "pending", "--url", "http://svc")
    captured: list[list[str]] = []
    monkeypatch.setattr(cli_module, "main", lambda argv: captured.append(list(argv)) or 3)

    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    assert excinfo.value.code == 3
    assert captured == [["service", "pending", "--url", "http://svc"]]
    assert runtime["prepare"] == []


@pytest.mark.parametrize("name", ["agent-gym", "gym", "ng"])
def test_every_canonical_name_intercepts_project_commands(
    name: str, monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]
) -> None:
    _argv(monkeypatch, name, "component", "run", "benchmarks/x.py")
    captured: list[list[str]] = []
    monkeypatch.setattr(cli_module, "main", lambda argv: captured.append(list(argv)) or 0)

    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    assert excinfo.value.code == 0
    assert captured == [["component", "run", "benchmarks/x.py"]]


def test_legacy_entrypoint_does_not_intercept_project_commands(
    monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]
) -> None:
    _argv(monkeypatch, "nemo_gym_eval", "service")
    monkeypatch.setattr(cli_module, "main", lambda argv: pytest.fail("legacy entry point must not dispatch"))
    captured: list[list[str]] = []
    monkeypatch.setattr(legacy_module, "main", lambda: captured.append(list(launcher.sys.argv)))

    launcher.main()

    assert captured == [["nemo_gym_eval", "service"]]
    assert runtime["prepare"] == [{"require_overlay": True}]


def test_probe_env_short_circuits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _argv(monkeypatch, "agent-gym", "service")
    monkeypatch.setenv(launcher.PROBE_ENV, "1")
    launcher.main()
    assert capsys.readouterr().out == "agent-gym:agent-gym\n"


@pytest.mark.parametrize(
    ("argv", "expected_topic"),
    [
        (("agent-gym", "help"), []),
        (("agent-gym", "help", "service"), ["service"]),
        (("gym", "-h"), []),
        (("ng", "--help"), []),
    ],
)
def test_top_level_help_runs_without_overlay(
    argv: tuple[str, ...],
    expected_topic: list[str],
    monkeypatch: pytest.MonkeyPatch,
    runtime: dict[str, list],
) -> None:
    _argv(monkeypatch, *argv)
    captured: list[list[str]] = []
    monkeypatch.setattr(cli_module, "print_help", lambda args=(): captured.append(list(args)) or 0)
    monkeypatch.setattr(upstream_main_module, "main", lambda: pytest.fail("help must not delegate upstream"))

    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    assert excinfo.value.code == 0
    assert captured == [expected_topic]
    assert runtime["prepare"] == [{"require_overlay": False}]


def test_upstream_subcommand_help_is_untouched(
    monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]
) -> None:
    _argv(monkeypatch, "agent-gym", "eval", "prepare", "--help")
    monkeypatch.setattr(cli_module, "print_help", lambda args=(): pytest.fail("only the first token help is ours"))
    captured: list[list[str]] = []
    monkeypatch.setattr(upstream_main_module, "main", lambda: captured.append(list(launcher.sys.argv)))

    launcher.main()

    assert captured == [["agent-gym", "eval", "prepare", "--help"]]
    assert runtime["prepare"] == [{"require_overlay": False}]


def test_version_stays_upstream(monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]) -> None:
    _argv(monkeypatch, "agent-gym", "--version")
    monkeypatch.setattr(cli_module, "print_help", lambda args=(): pytest.fail("--version is upstream"))
    captured: list[list[str]] = []
    monkeypatch.setattr(upstream_main_module, "main", lambda: captured.append(list(launcher.sys.argv)))

    launcher.main()

    assert captured == [["agent-gym", "--version"]]
    assert runtime["prepare"] == [{"require_overlay": False}]


def test_doctor_is_still_intercepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _argv(monkeypatch, "agent-gym", "doctor", "--json")
    captured: list[list[str]] = []
    monkeypatch.setattr(doctor, "main", lambda argv: captured.append(list(argv)) or 4)

    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    assert excinfo.value.code == 4
    assert captured == [["--json"]]


def test_non_diagnostic_invocation_prepares_overlay_and_staging(
    monkeypatch: pytest.MonkeyPatch, runtime: dict[str, list]
) -> None:
    _argv(monkeypatch, "agent-gym", "list")
    monkeypatch.setattr(launcher, "_needs_prepare_staging", lambda name, argv: True)
    captured: list[list[str]] = []
    monkeypatch.setattr(upstream_main_module, "main", lambda: captured.append(list(launcher.sys.argv)))

    launcher.main()

    assert captured == [["agent-gym", "list"]]
    assert runtime["prepare"] == [{"require_overlay": True}]
    assert runtime["stage"] == ["layout"]
    assert runtime["activate"] == ["layout"]
    assert runtime["cleanup"] == ["layout", "layout"]
