from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from agent_gym import cli


class FakeResponse:
    def __init__(self, body: bytes = b"{}", status: int = 200) -> None:
        self.body = body
        self.status = status

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class FakeOpener:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request: urllib.request.Request, timeout: float):
        self.requests.append((request, timeout))
        response = self.responses.pop(0) if self.responses else FakeResponse()
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture(autouse=True)
def _stub_upstream_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_upstream_help_text", lambda: "UPSTREAM HELP\n")


def _body(request: urllib.request.Request) -> Any:
    assert request.data is not None
    return json.loads(request.data.decode("utf-8"))


def test_top_level_help_combines_upstream_and_project_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "UPSTREAM HELP" in out
    assert "AgentGym commands:" in out
    assert "prepare-data" in out


def test_help_subcommand_and_topics(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["help"]) == 0
    assert "AgentGym commands:" in capsys.readouterr().out

    assert cli.main(["help", "service"]) == 0
    assert "agent-gym service <command>" in capsys.readouterr().out

    assert cli.main(["help", "prepare-data"]) == 0
    assert "Usage: agent-gym prepare-data" in capsys.readouterr().out

    assert cli.main(["help", "bogus"]) == 2
    assert "unknown help topic: bogus" in capsys.readouterr().err


def test_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown agent-gym command: bogus" in err
    assert "agent-gym help" in err


def test_service_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["service"]) == 2
    assert "agent-gym service <command>" in capsys.readouterr().err

    assert cli.main(["service", "--help"]) == 0
    assert "agent-gym service <command>" in capsys.readouterr().out


def test_service_evaluate_posts_full_payload(capsys: pytest.CaptureFixture[str]) -> None:
    opener = FakeOpener([FakeResponse(json.dumps({"eval_id": "e1", "status": "queued"}).encode())])

    code = cli.main(
        [
            "service",
            "evaluate",
            "--candidate-id",
            "cand-1",
            "--candidate-digest",
            "sha256:" + "a" * 64,
            "--adapter",
            "dsh",
            "--benchmark",
            "gsm8k",
            "--split",
            "train_probe",
            "--tier",
            "probe",
            "--seed",
            "42",
            "--harness-manifest",
            "v1",
            "--dataset-version",
            "auto",
            "--task-id",
            "t1",
            "--task-id",
            "t2",
            "--budget-timeout-s",
            "600",
            "--budget-max-rollouts",
            "8",
            "--acknowledge-exhausted-split",
            "--acknowledge-lineage-budget",
            "--url",
            "http://example.test:8399/",
            "--timeout",
            "5",
        ],
        opener=opener,
    )

    assert code == 0
    request, timeout = opener.requests[0]
    assert request.full_url == "http://example.test:8399/gym_evaluate"
    assert request.get_method() == "POST"
    assert timeout == 5.0
    assert _body(request) == {
        "candidate_id": "cand-1",
        "candidate_digest": "sha256:" + "a" * 64,
        "adapter": "dsh",
        "benchmark": "gsm8k",
        "split": "train_probe",
        "tier": "probe",
        "seed": 42,
        "harness_manifest": "v1",
        "dataset_version": "auto",
        "task_ids": ["t1", "t2"],
        "budget": {"timeout_s": 600, "max_rollouts": 8},
        "acknowledge_exhausted_split": True,
        "acknowledge_lineage_budget": True,
    }
    output = json.loads(capsys.readouterr().out)
    assert output == {"eval_id": "e1", "status": "queued"}


def test_service_evaluate_minimal_payload_and_default_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cli.SERVICE_URL_ENV, "http://env.test:1234/")
    opener = FakeOpener()

    code = cli.main(
        [
            "service",
            "evaluate",
            "--candidate-id",
            "c",
            "--adapter",
            "a",
            "--benchmark",
            "b",
            "--split",
            "s",
            "--tier",
            "full",
            "--seed",
            "0",
        ],
        opener=opener,
    )

    assert code == 0
    request, timeout = opener.requests[0]
    assert request.full_url == "http://env.test:1234/gym_evaluate"
    assert timeout == 30.0
    assert _body(request) == {
        "candidate_id": "c",
        "adapter": "a",
        "benchmark": "b",
        "split": "s",
        "tier": "full",
        "seed": 0,
    }
    assert capsys.readouterr().out == "{}\n"


def test_service_client_leaf_payloads() -> None:
    cases = [
        (["result", "--eval-id", "e1"], "/results", {"eval_id": "e1"}),
        (["pending"], "/pending", {}),
        (["tasks", "--benchmark", "gsm8k", "--split", "train"], "/tasks", {"benchmark": "gsm8k", "split": "train"}),
        (["feedback", "--candidate-id", "c1"], "/feedback", {"candidate_id": "c1"}),
    ]
    for argv, path, expected in cases:
        opener = FakeOpener()
        assert cli.main(["service", *argv, "--url", "http://svc:8399"], opener=opener) == 0
        request, _ = opener.requests[0]
        assert request.full_url == f"http://svc:8399{path}"
        assert request.get_method() == "POST"
        assert _body(request) == expected


def test_promotion_submit_payload_and_seed_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "service",
                "promotion",
                "submit",
                "--candidate-id",
                "c1",
                "--candidate-digest",
                "sha256:" + "a" * 64,
                "--adapter",
                "dsh",
                "--benchmark",
                "gaia",
            ]
        )
    assert excinfo.value.code == 2
    capsys.readouterr()

    opener = FakeOpener([FakeResponse(b'{"promotion_id": "p1"}')])
    assert (
        cli.main(
            [
                "service",
                "promotion",
                "submit",
                "--candidate-id",
                "c1",
                "--candidate-digest",
                "sha256:" + "a" * 64,
                "--adapter",
                "dsh",
                "--benchmark",
                "gaia",
                "--seed",
                "7",
                "--budget-timeout-s",
                "60",
            ],
            opener=opener,
        )
        == 0
    )
    request, _ = opener.requests[0]
    assert request.full_url == f"{cli.DEFAULT_SERVICE_URL}/promotion"
    assert _body(request) == {
        "candidate_id": "c1",
        "candidate_digest": "sha256:" + "a" * 64,
        "adapter": "dsh",
        "benchmark": "gaia",
        "seed": 7,
        "tier": "full",
        "budget": {"timeout_s": 60},
    }


def test_promotion_rejects_non_full_tier(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "service",
                "promotion",
                "submit",
                "--candidate-id",
                "c1",
                "--candidate-digest",
                "sha256:" + "a" * 64,
                "--adapter",
                "dsh",
                "--benchmark",
                "gaia",
                "--seed",
                "7",
                "--tier",
                "probe",
            ]
        )
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_promotion_status_uses_get_and_escapes_id() -> None:
    opener = FakeOpener([FakeResponse(b'{"status": "running"}')])
    assert (
        cli.main(
            ["service", "promotion", "status", "--promotion-id", "a/b c", "--url", "http://svc"],
            opener=opener,
        )
        == 0
    )
    request, _ = opener.requests[0]
    assert request.full_url == "http://svc/promotion/a%2Fb%20c"
    assert request.get_method() == "GET"
    assert request.data is None


def test_http_error_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    error = urllib.error.HTTPError(
        "http://svc/pending",
        400,
        "Bad Request",
        hdrs=None,
        fp=io.BytesIO(b'{"detail": "nope"}'),
    )
    assert cli.main(["service", "pending", "--url", "http://svc"], opener=FakeOpener([error])) == 1
    err = capsys.readouterr().err
    assert "HTTP 400" in err
    assert '"detail": "nope"' in err


def test_connection_error_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    error = urllib.error.URLError("connection refused")
    assert cli.main(["service", "pending", "--url", "http://127.0.0.1:1"], opener=FakeOpener([error])) == 1
    assert "cannot reach http://127.0.0.1:1" in capsys.readouterr().err


def test_invalid_url_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["service", "pending", "--url", "not-a-url"], opener=FakeOpener()) == 2
    assert "invalid service URL not-a-url" in capsys.readouterr().err


def test_non_positive_timeout_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["service", "pending", "--url", "http://svc", "--timeout", "0"], opener=FakeOpener()) == 2
    assert "--timeout must be a positive" in capsys.readouterr().err


def test_non_json_response_is_printed_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["service", "pending"], opener=FakeOpener([FakeResponse(b"plain text\n")])) == 0
    assert capsys.readouterr().out == "plain text\n"

    assert cli.main(["service", "pending"], opener=FakeOpener([FakeResponse(b"")])) == 0
    assert capsys.readouterr().out == ""


def test_release_wraps_script_from_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "build.py"
    script.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(cli, "_require_source_file", lambda relative, *, command: script)
    monkeypatch.setattr(cli, "_run_script", lambda command: calls.append(list(command)) or 9)

    assert cli.main(["release", "build", "--check"]) == 9
    assert calls == [[cli.sys.executable, str(script), "--check"]]
    capsys.readouterr()

    assert cli.main(["release"]) == 2
    assert "usage: agent-gym release" in capsys.readouterr().err

    assert cli.main(["release", "bogus"]) == 2
    assert "unknown release action: bogus" in capsys.readouterr().err


def test_lint_wraps_shell_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "lint.sh"
    script.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(cli, "_require_source_file", lambda relative, *, command: script)
    monkeypatch.setattr(cli, "_run_script", lambda command: calls.append(list(command)) or 0)

    assert cli.main(["lint"]) == 0
    assert calls == [["bash", str(script)]]


def test_require_source_file_reports_wheel_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_repo_roots", lambda layout=None: [tmp_path])
    assert cli._require_source_file("scripts/ci/lint.sh", command="agent-gym lint") is None
    err = capsys.readouterr().err
    assert "available only from a source checkout" in err
    assert str(tmp_path / "scripts/ci/lint.sh") in err


def test_repo_roots_dedupes_and_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_GYM_REPO", str(tmp_path))
    monkeypatch.setattr(cli, "GymLayout", _LayoutStub(tmp_path))
    roots = cli._repo_roots()
    assert roots[0] == tmp_path.resolve()
    assert len(roots) == len(set(roots))


class _LayoutStub:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def discover(self) -> Any:
        return type("L", (), {"repo_root": self.repo})()


def test_component_forwards_run_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_main(argv):
        captured.append(list(argv))
        return 5

    monkeypatch.setattr(cli.component, "main", fake_main)
    assert cli.main(["component", "run", "benchmarks/x.py", "--", "--flag"]) == 5
    assert captured == [["benchmarks/x.py", "--", "--flag"]]

    assert cli.main(["component", "bogus"]) == 2


def test_prepare_data_requires_source_and_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "scripts" / "prepare_data.sh"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(cli, "_require_source_file", lambda relative, *, command: script)

    def fake_main(argv, *, layout=None):
        captured["argv"] = list(argv)
        captured["repo"] = layout.repo_root
        return 4

    monkeypatch.setattr(cli.prepare_data, "main", fake_main)
    assert cli.main(["prepare-data", "--dry-run", "all"]) == 4
    assert captured["argv"] == ["--dry-run", "all"]
    assert captured["repo"] == tmp_path.resolve()

    monkeypatch.setattr(cli, "_require_source_file", lambda relative, *, command: None)
    assert cli.main(["prepare-data", "all"]) == 2
