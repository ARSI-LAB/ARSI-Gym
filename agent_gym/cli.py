"""AgentGym project commands layered on top of the upstream nemo-gym CLI.

The launcher intercepts these commands before ``prepare_runtime`` so relative
paths keep the caller's cwd. Each leaf prepares whatever runtime it needs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from agent_gym import component, prepare_data
from agent_gym.layout import GymLayout


SERVICE_URL_ENV = "AGENT_GYM_SERVICE_URL"
DEFAULT_SERVICE_URL = "http://127.0.0.1:8399"

PROJECT_COMMANDS = ("service", "prepare-data", "component", "release", "lint")

RELEASE_SCRIPTS = {
    "build": "scripts/refactor/build_release_artifacts.py",
    "install": "scripts/refactor/install_release.py",
    "stage": "scripts/refactor/stage_upgrade_tree.py",
    "audit": "scripts/refactor/audit_tree.py",
    "manifest": "scripts/refactor/update_overlay_manifest.py",
}

Opener = Callable[..., Any]


def _project_help_text() -> str:
    return (
        "\nAgentGym commands:\n"
        "  service        Start the evaluation service or call its HTTP API.\n"
        "  prepare-data   Rebuild benchmark artifacts (prepare/collate/render/rsi/all).\n"
        "  component      Run a component script from the staged workspace.\n"
        "  release        Build/install/stage/audit release artifacts and refresh the manifest.\n"
        "  lint           Run the repository lint suite.\n"
        "\n"
        "Run `agent-gym help <command>` for command-specific help.\n"
    )


def _upstream_help_text() -> str:
    try:
        from nemo_gym.cli.main import build_parser
    except Exception:  # pragma: no cover - upstream is a hard dependency, but help must never crash
        return "usage: agent-gym [-h] [--version] {list,search,dataset,env,eval,dev} ...\n"
    return build_parser().format_help()


def _top_level_help_text() -> str:
    upstream = _upstream_help_text().rstrip("\n")
    return f"{upstream}\n{_project_help_text()}"


def _service_usage() -> str:
    return (
        "usage: agent-gym service <command> [options]\n"
        "\n"
        "Commands:\n"
        "  run [SERVER FLAGS]        Start the evaluation service (HTTP + MCP).\n"
        "  evaluate                  Submit one evaluation request.\n"
        "  result --eval-id ID       Fetch the result for an eval id.\n"
        "  pending                   List queued and running eval ids.\n"
        "  tasks --benchmark B --split S\n"
        "                            List task ids for a registered benchmark split.\n"
        "  feedback --candidate-id ID\n"
        "                            Read a candidate's promotion feedback summary.\n"
        "  promotion submit          Enqueue a locked holdout promotion run.\n"
        "  promotion status --promotion-id ID\n"
        "                            Poll one promotion report.\n"
        "\n"
        f"The default service URL is ${SERVICE_URL_ENV} or {DEFAULT_SERVICE_URL}.\n"
        "Every client command accepts --url URL and --timeout SECONDS.\n"
    )


def _release_usage() -> str:
    actions = "|".join(RELEASE_SCRIPTS)
    return (
        f"usage: agent-gym release {{{actions}}} [PASSTHROUGH ...]\n"
        "\n"
        "Run the matching scripts/refactor/*.py helper with the caller's cwd and\n"
        "forward its exit code. Requires a source checkout.\n"
    )


def _lint_usage() -> str:
    return (
        "usage: agent-gym lint [PASSTHROUGH ...]\n"
        "\n"
        "Run scripts/ci/lint.sh with the caller's cwd and forward its exit code.\n"
        "Requires a source checkout.\n"
    )


def print_help(argv: Sequence[str] = (), *, stream=None) -> int:
    arguments = list(argv)
    target = sys.stdout if stream is None else stream
    if arguments:
        command = arguments[0]
        help_text = {
            "service": _service_usage,
            "prepare-data": prepare_data.usage,
            "component": component.usage,
            "release": _release_usage,
            "lint": _lint_usage,
        }.get(command)
        if help_text is None:
            print(f"error: unknown help topic: {command}", file=sys.stderr)
            return 2
        text = help_text()
    else:
        text = _top_level_help_text()
    print(text, file=target, end="" if text.endswith("\n") else "\n")
    return 0


def _repo_roots(layout: GymLayout | None = None) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("AGENT_GYM_REPO")
    if configured:
        candidates.append(Path(configured).expanduser().resolve(strict=False))
    candidates.append((layout if layout is not None else GymLayout.discover()).repo_root)
    candidates.append(Path(__file__).resolve().parents[1])
    roots: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _require_source_file(relative: str, *, command: str, layout: GymLayout | None = None) -> Path | None:
    roots = _repo_roots(layout)
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    print(f"error: {command} is available only from a source checkout; missing {relative}", file=sys.stderr)
    print("checked:", file=sys.stderr)
    for root in roots:
        print(f"  - {root / relative}", file=sys.stderr)
    return None


def _run_script(command: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> int:
    return runner(list(command), cwd=Path.cwd(), check=False).returncode


def _service_url(value: str | None) -> str:
    return (value or os.environ.get(SERVICE_URL_ENV) or DEFAULT_SERVICE_URL).rstrip("/")


def _read_body(response: Any) -> str:
    body = response.read()
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _http_json(
    url: str,
    payload: Any = None,
    *,
    method: str = "POST",
    timeout: float = 30.0,
    opener: Opener | None = None,
) -> int:
    if timeout <= 0:
        print("error: --timeout must be a positive number of seconds", file=sys.stderr)
        return 2
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        request = urllib.request.Request(url, data=data, method=method)
    except ValueError as exc:
        print(f"error: invalid service URL {url}: {exc}", file=sys.stderr)
        return 2
    if data is not None:
        request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    open_request = urllib.request.urlopen if opener is None else opener
    try:
        response = open_request(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        print(f"error: HTTP {exc.code} from {url}", file=sys.stderr)
        body = _read_body(exc)
        if body.strip():
            print(body.rstrip("\n"), file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: cannot reach {url}: {exc}", file=sys.stderr)
        return 1
    with response:
        text = _read_body(response)
    if not text.strip():
        return 0
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(text, end="" if text.endswith("\n") else "\n")
        return 0
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return 0


def _client_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--url", default=None, help=f"service URL (default: ${SERVICE_URL_ENV} or {DEFAULT_SERVICE_URL})")
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    return parser


def _budget_payload(args: argparse.Namespace) -> dict[str, int]:
    budget: dict[str, int] = {}
    if args.budget_timeout_s is not None:
        budget["timeout_s"] = args.budget_timeout_s
    if args.budget_max_rollouts is not None:
        budget["max_rollouts"] = args.budget_max_rollouts
    return budget


def _add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget-timeout-s", type=int, default=None)
    parser.add_argument("--budget-max-rollouts", type=int, default=None)


def _service_evaluate(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service evaluate", "Submit one evaluation request.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-digest", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--tier", required=True, choices=["probe", "screen", "full"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harness-manifest", default=None)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--task-id", dest="task_ids", action="append", default=None)
    parser.add_argument("--acknowledge-exhausted-split", action="store_true")
    parser.add_argument("--acknowledge-lineage-budget", action="store_true")
    _add_budget_arguments(parser)
    args = parser.parse_args(list(argv))

    payload: dict[str, Any] = {
        "candidate_id": args.candidate_id,
        "adapter": args.adapter,
        "benchmark": args.benchmark,
        "split": args.split,
        "tier": args.tier,
        "seed": args.seed,
    }
    if args.candidate_digest is not None:
        payload["candidate_digest"] = args.candidate_digest
    if args.harness_manifest is not None:
        payload["harness_manifest"] = args.harness_manifest
    if args.dataset_version is not None:
        payload["dataset_version"] = args.dataset_version
    if args.task_ids is not None:
        payload["task_ids"] = args.task_ids
    budget = _budget_payload(args)
    if budget:
        payload["budget"] = budget
    if args.acknowledge_exhausted_split:
        payload["acknowledge_exhausted_split"] = True
    if args.acknowledge_lineage_budget:
        payload["acknowledge_lineage_budget"] = True
    return _http_json(f"{_service_url(args.url)}/gym_evaluate", payload, timeout=args.timeout, opener=opener)


def _service_result(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service result", "Fetch the result for an eval id.")
    parser.add_argument("--eval-id", required=True)
    args = parser.parse_args(list(argv))
    return _http_json(
        f"{_service_url(args.url)}/results", {"eval_id": args.eval_id}, timeout=args.timeout, opener=opener
    )


def _service_pending(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service pending", "List queued and running eval ids.")
    args = parser.parse_args(list(argv))
    return _http_json(f"{_service_url(args.url)}/pending", {}, timeout=args.timeout, opener=opener)


def _service_tasks(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service tasks", "List task ids for a registered benchmark split.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args(list(argv))
    return _http_json(
        f"{_service_url(args.url)}/tasks",
        {"benchmark": args.benchmark, "split": args.split},
        timeout=args.timeout,
        opener=opener,
    )


def _service_feedback(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service feedback", "Read a candidate's promotion feedback summary.")
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args(list(argv))
    return _http_json(
        f"{_service_url(args.url)}/feedback",
        {"candidate_id": args.candidate_id},
        timeout=args.timeout,
        opener=opener,
    )


def _promotion_submit(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service promotion submit", "Enqueue a locked holdout promotion run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-digest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tier", default="full", choices=["full"])
    parser.add_argument("--dataset-version", default=None)
    _add_budget_arguments(parser)
    args = parser.parse_args(list(argv))

    payload: dict[str, Any] = {
        "candidate_id": args.candidate_id,
        "candidate_digest": args.candidate_digest,
        "adapter": args.adapter,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "tier": args.tier,
    }
    if args.dataset_version is not None:
        payload["dataset_version"] = args.dataset_version
    budget = _budget_payload(args)
    if budget:
        payload["budget"] = budget
    return _http_json(f"{_service_url(args.url)}/promotion", payload, timeout=args.timeout, opener=opener)


def _promotion_status(argv: Sequence[str], *, opener: Opener | None) -> int:
    parser = _client_parser("agent-gym service promotion status", "Poll one promotion report.")
    parser.add_argument("--promotion-id", required=True)
    args = parser.parse_args(list(argv))
    promotion_id = urllib.parse.quote(args.promotion_id, safe="")
    return _http_json(
        f"{_service_url(args.url)}/promotion/{promotion_id}",
        None,
        method="GET",
        timeout=args.timeout,
        opener=opener,
    )


def _promotion_main(argv: Sequence[str], *, opener: Opener | None) -> int:
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(_service_usage(), file=stream, end="")
        return 0 if arguments else 2
    action = arguments[0]
    handlers = {"submit": _promotion_submit, "status": _promotion_status}
    handler = handlers.get(action)
    if handler is None:
        print(f"error: unknown promotion action: {action}", file=sys.stderr)
        print(_service_usage(), file=sys.stderr, end="")
        return 2
    return handler(arguments[1:], opener=opener)


def _service_main(argv: Sequence[str], *, opener: Opener | None = None) -> int:
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(_service_usage(), file=stream, end="")
        return 0 if arguments else 2
    command = arguments[0]
    rest = arguments[1:]
    if command == "run":
        from eval_service.server import main as server_main

        return server_main(rest, prog="agent-gym service run")
    if command == "promotion":
        return _promotion_main(rest, opener=opener)
    handlers = {
        "evaluate": _service_evaluate,
        "result": _service_result,
        "pending": _service_pending,
        "tasks": _service_tasks,
        "feedback": _service_feedback,
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"error: unknown service command: {command}", file=sys.stderr)
        print(_service_usage(), file=sys.stderr, end="")
        return 2
    return handler(rest, opener=opener)


def _prepare_data_main(argv: Sequence[str]) -> int:
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(prepare_data.usage(), file=stream, end="")
        return 0 if arguments else 2
    script = _require_source_file("scripts/prepare_data.sh", command="agent-gym prepare-data")
    if script is None:
        return 2
    layout = GymLayout.discover(repo_root=script.parent.parent)
    return prepare_data.main(arguments, layout=layout)


def _component_main(argv: Sequence[str]) -> int:
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(component.usage(), file=stream, end="")
        return 0 if arguments else 2
    if arguments[0] != "run":
        print(f"error: unknown component action: {arguments[0]}", file=sys.stderr)
        print(component.usage(), file=sys.stderr, end="")
        return 2
    return component.main(arguments[1:])


def _release_main(argv: Sequence[str]) -> int:
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        stream = sys.stdout if arguments else sys.stderr
        print(_release_usage(), file=stream, end="")
        return 0 if arguments else 2
    action = arguments[0]
    relative = RELEASE_SCRIPTS.get(action)
    if relative is None:
        print(f"error: unknown release action: {action}", file=sys.stderr)
        print(_release_usage(), file=sys.stderr, end="")
        return 2
    script = _require_source_file(relative, command=f"agent-gym release {action}")
    if script is None:
        return 2
    return _run_script([sys.executable, str(script), *arguments[1:]])


def _lint_main(argv: Sequence[str]) -> int:
    arguments = list(argv)
    if arguments and arguments[0] in {"-h", "--help"}:
        print(_lint_usage(), end="")
        return 0
    script = _require_source_file("scripts/ci/lint.sh", command="agent-gym lint")
    if script is None:
        return 2
    return _run_script(["bash", str(script), *arguments])


def main(argv: Sequence[str] | None = None, *, opener: Opener | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return print_help()
    command = arguments[0]
    rest = arguments[1:]
    if command in {"help", "-h", "--help"}:
        return print_help(rest)
    if command == "service":
        return _service_main(rest, opener=opener)
    if command == "prepare-data":
        return _prepare_data_main(rest)
    if command == "component":
        return _component_main(rest)
    if command == "release":
        return _release_main(rest)
    if command == "lint":
        return _lint_main(rest)
    print(f"error: unknown agent-gym command: {command}", file=sys.stderr)
    print("Run `agent-gym help` for available commands.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
