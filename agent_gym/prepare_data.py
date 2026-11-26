"""Rebuild benchmark artifacts that are intentionally not tracked by Git.

This is the Python implementation behind ``agent-gym prepare-data``. It mirrors
``scripts/prepare_data.sh`` so both entry points stay behaviorally identical.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from agent_gym.layout import OVERLAY_MANIFEST, GymLayout, LayoutError


ALL_BENCHMARKS = ("aegis_v4_safety", "arc-agi", "arc-agi-v2", "arc-agi-v3", "gaia", "gsm8k")
RENDERABLE_BENCHMARKS = ("arc-agi", "arc-agi-v2", "gsm8k")
COLLATEABLE_BENCHMARKS = ("aegis_v4_safety", "arc-agi", "arc-agi-v2", "gsm8k")
RSI_BENCHMARKS = ("arc-agi", "gaia")

JSONL_PATHS = {
    "aegis_v4_safety": "benchmarks/aegis_v4_safety/data/aegis_v4_safety_benchmark.jsonl",
    "arc-agi": "benchmarks/arc-agi/data/arc_agi_benchmark.jsonl",
    "arc-agi-v2": "benchmarks/arc-agi-v2/data/arc_agi_v2_benchmark.jsonl",
    "arc-agi-v3": "benchmarks/arc-agi-v3/data/arc_agi_v3_games.jsonl",
    "gaia": "benchmarks/gaia/data/gaia_text_validation.jsonl",
    "gsm8k": "benchmarks/gsm8k/data/gsm8k_benchmark.jsonl",
}

PROMPT_CONFIGS = {
    "arc-agi": "benchmarks/prompts/generic/arc.yaml",
    "arc-agi-v2": "benchmarks/prompts/generic/arc.yaml",
    "gsm8k": "benchmarks/prompts/generic/math.yaml",
}

RSI_SOURCES = {
    "arc-agi": "benchmarks/arc-agi/data/arc_agi_benchmark_materialized.jsonl",
    "gaia": "benchmarks/gaia/data/gaia_text_validation.jsonl",
}

SPLITTER = "benchmarks/rsi_dataset_tools/split_jsonl.py"
RUN_COMPONENT = "scripts/run_component.py"

Runner = Callable[..., subprocess.CompletedProcess[str]]


def usage() -> str:
    return (
        "Usage: agent-gym prepare-data [--dry-run] <command> [name ...]\n"
        "       bash scripts/prepare_data.sh [--dry-run] <command> [name ...]\n"
        "\n"
        "Commands:\n"
        "  prepare [name ...]  Run `gym eval prepare` for the named benchmark(s).\n"
        "  collate [name ...]  Write metrics + *_prepare.jsonl where upstream collate is compatible.\n"
        "  render  [name ...]  Materialize prompt-ready rows for ARC-AGI/ARC-AGI-2/GSM8K.\n"
        "  rsi     [name ...]  Recreate the ARC-AGI and GAIA RSI train/holdout splits.\n"
        "  all                  Run prepare, collate, render and rsi for all supported benchmarks.\n"
        "\n"
        "  `collate` supports aegis_v4_safety, arc-agi, arc-agi-v2 and gsm8k. GAIA rows already\n"
        "  carry responses_create_params.input, and ARC-AGI-3 has a self-contained agent config;\n"
        "  both use their prepare output directly instead of passing through this collate step.\n"
        "\n"
        "Names:\n"
        "  all, aegis_v4_safety, arc-agi, arc-agi-v2, arc-agi-v3, gaia, gsm8k\n"
        "\n"
        "Environment:\n"
        "  GYM_BIN                 Command to invoke (default: agent-gym).\n"
        "  PYTHON_BIN              Python used for split/resolve helpers.\n"
        "  AGENT_GYM_PREPARE_OUTPUT\n"
        "                          Collate output directory (default: .tmp/data-prep).\n"
        "\n"
        "Notes:\n"
        "  * No benchmark or component data is tracked in Git. `prepare arc-agi` also\n"
        "    regenerates the base ARC example fixture used by arc_agi_simple_agent.\n"
        "  * `all` may download public datasets and GAIA requires prior Hugging Face access.\n"
        "  * pytest and `gym env start` do not prepare benchmark data automatically.\n"
    )


@dataclass(frozen=True)
class Context:
    layout: GymLayout
    repo_root: Path
    overlay_root: Path
    output_root: Path
    gym_bin: str
    python_bin: str
    dry_run: bool
    environ: Mapping[str, str]


def _default_layout() -> GymLayout:
    layout = GymLayout.discover()
    if (layout.overlay_root / OVERLAY_MANIFEST).is_file():
        return layout
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "overlay" / OVERLAY_MANIFEST).is_file():
        return GymLayout.discover(repo_root=source_root)
    return layout


def _resolve_relative(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _context(layout: GymLayout | None, environ: Mapping[str, str], *, dry_run: bool) -> Context:
    active_layout = layout if layout is not None else _default_layout()
    repo_root = active_layout.repo_root
    overlay_root = active_layout.overlay_root
    output_value = environ.get("AGENT_GYM_PREPARE_OUTPUT")
    output_root = (
        _resolve_relative(output_value, repo_root)
        if output_value
        else repo_root / ".tmp" / "data-prep"
    )
    repo_value = environ.get("AGENT_GYM_REPO") or str(repo_root)
    overlay_value = environ.get("AGENT_GYM_OVERLAY") or str(overlay_root)
    child_environ = dict(environ)
    child_environ["AGENT_GYM_REPO"] = repo_value
    child_environ["AGENT_GYM_OVERLAY"] = overlay_value
    pythonpath = [overlay_value, repo_value]
    existing = environ.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    child_environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return Context(
        layout=active_layout,
        repo_root=repo_root,
        overlay_root=overlay_root,
        output_root=output_root,
        gym_bin=environ.get("GYM_BIN") or "agent-gym",
        python_bin=environ.get("PYTHON_BIN") or sys.executable,
        dry_run=dry_run,
        environ=child_environ,
    )


def _die(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _run(ctx: Context, command: Sequence[str], runner: Runner) -> int:
    print("+ " + " ".join(shlex.quote(part) for part in command))
    if ctx.dry_run:
        return 0
    return runner(list(command), cwd=ctx.repo_root, env=dict(ctx.environ), check=False).returncode


def expand_targets(defaults: Sequence[str], names: Sequence[str]) -> list[str]:
    if not names:
        return list(defaults)
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        for target in (defaults if name == "all" else (name,)):
            if target in seen:
                continue
            seen.add(target)
            result.append(target)
    return result


def _seed_arc_agi_example(ctx: Context, runner: Runner) -> int:
    relative = "resources_servers/arc_agi/data/example.jsonl"
    if (ctx.repo_root / relative).is_file():
        return 0
    if ctx.dry_run:
        print(f"+ generate {relative} with resources_servers/arc_agi/prepare_example.py")
        return 0
    return _run(ctx, [ctx.python_bin, RUN_COMPONENT, "resources_servers/arc_agi/prepare_example.py"], runner)


def _seed_prepare_input(ctx: Context, benchmark: str, runner: Runner) -> int:
    if benchmark == "arc-agi":
        return _seed_arc_agi_example(ctx, runner)
    if benchmark != "aegis_v4_safety":
        return 0

    relative = "benchmarks/aegis_v4_safety/data/source.jsonl"
    destination = ctx.repo_root / relative
    if destination.is_file():
        return 0
    if ctx.dry_run:
        print(f"+ seed {relative} from the pinned wheel when missing")
        return 0

    try:
        source = ctx.layout.resolve_data(relative)
    except LayoutError:
        return _die(f"cannot resolve {relative} from overlay/repo/wheel")
    if not source.is_file():
        return _die(f"cannot resolve {relative} from overlay/repo/wheel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"Seeded {relative} from {source}")
    return 0


def _prepare_benchmarks(ctx: Context, targets: Sequence[str], runner: Runner) -> int:
    for benchmark in targets:
        if benchmark not in ALL_BENCHMARKS:
            return _die(f"unknown benchmark: {benchmark}")
        code = _seed_prepare_input(ctx, benchmark, runner)
        if code:
            return code
        # One benchmark per process: v1/v2 share a resources-server path and upstream
        # deduplicates them, silently preparing only one ARC variant.
        code = _run(ctx, [ctx.gym_bin, "eval", "prepare", "--benchmark", benchmark], runner)
        if code:
            return code
    return 0


def _collate_benchmarks(ctx: Context, targets: Sequence[str], runner: Runner) -> int:
    for benchmark in targets:
        if benchmark not in ALL_BENCHMARKS:
            return _die(f"unknown benchmark: {benchmark}")
        if benchmark not in COLLATEABLE_BENCHMARKS:
            return _die(
                f"collate is not supported for {benchmark}; use prepare output directly "
                "(GAIA already carries model input; ARC-AGI-3 is self-contained)"
            )
        relative = JSONL_PATHS[benchmark]
        data_path = ctx.repo_root / relative
        if not data_path.is_file() and not ctx.dry_run:
            return _die(f"missing {relative}; run: bash scripts/prepare_data.sh prepare {benchmark}")
        code = _run(
            ctx,
            [
                ctx.gym_bin,
                "dataset",
                "collate",
                "--config",
                f"benchmarks/{benchmark}/config.yaml",
                "--output-dir",
                str(ctx.output_root / benchmark),
                "--mode",
                "train_preparation",
            ],
            runner,
        )
        if code:
            return code
    return 0


def _render_benchmarks(ctx: Context, targets: Sequence[str], runner: Runner) -> int:
    for benchmark in targets:
        if benchmark not in RENDERABLE_BENCHMARKS:
            return _die(
                f"render is not defined for {benchmark}; supported: {' '.join(RENDERABLE_BENCHMARKS)}"
            )
        relative = JSONL_PATHS[benchmark]
        data_path = ctx.repo_root / relative
        if not data_path.is_file() and not ctx.dry_run:
            return _die(f"missing {relative}; run: bash scripts/prepare_data.sh prepare {benchmark}")
        prompt_path = ctx.overlay_root / PROMPT_CONFIGS[benchmark]
        output_path = data_path.with_name(f"{data_path.stem}_materialized.jsonl")
        code = _run(
            ctx,
            [
                ctx.gym_bin,
                "dataset",
                "render",
                "--input",
                str(data_path),
                "--prompt-config",
                str(prompt_path),
                "--output",
                str(output_path),
            ],
            runner,
        )
        if code:
            return code
    return 0


def _rsi_splits(ctx: Context, targets: Sequence[str], runner: Runner) -> int:
    for benchmark in targets:
        if benchmark not in RSI_BENCHMARKS:
            return _die(
                f"RSI split is not defined for {benchmark}; supported: {' '.join(RSI_BENCHMARKS)}"
            )
        source_path = ctx.repo_root / RSI_SOURCES[benchmark]
        output_dir = ctx.repo_root / "benchmarks" / benchmark / "data" / "rsi"
        if not source_path.is_file() and not ctx.dry_run:
            return _die(f"missing {source_path.relative_to(ctx.repo_root)}; run prepare and render first")
        slug = benchmark.replace("-", "_")
        code = _run(
            ctx,
            [
                ctx.python_bin,
                RUN_COMPONENT,
                SPLITTER,
                str(source_path),
                "--train",
                str(output_dir / f"{slug}_train.jsonl"),
                "--holdout",
                str(output_dir / f"{slug}_holdout.jsonl"),
                "--id-field",
                "task_id",
            ],
            runner,
        )
        if code:
            return code
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    layout: GymLayout | None = None,
    runner: Runner = subprocess.run,
    environ: Mapping[str, str] | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(usage(), file=sys.stderr, end="")
        return 2

    dry_run = False
    remaining: list[str] = []
    for argument in arguments:
        if argument in {"-h", "--help"}:
            print(usage(), end="")
            return 0
        if argument == "--dry-run":
            dry_run = True
        else:
            remaining.append(argument)

    if not remaining:
        print(usage(), file=sys.stderr, end="")
        return _die("unknown command: ")

    command = remaining[0]
    names = remaining[1:]
    ctx = _context(layout, dict(os.environ if environ is None else environ), dry_run=dry_run)

    if command == "prepare":
        return _prepare_benchmarks(ctx, expand_targets(ALL_BENCHMARKS, names), runner)
    if command == "collate":
        return _collate_benchmarks(ctx, expand_targets(COLLATEABLE_BENCHMARKS, names), runner)
    if command == "render":
        return _render_benchmarks(ctx, expand_targets(RENDERABLE_BENCHMARKS, names), runner)
    if command == "rsi":
        return _rsi_splits(ctx, expand_targets(RSI_BENCHMARKS, names), runner)
    if command == "all":
        if names:
            return _die("all does not accept benchmark names; run individual commands for a subset")
        for step in (
            lambda: _prepare_benchmarks(ctx, ALL_BENCHMARKS, runner),
            lambda: _collate_benchmarks(ctx, COLLATEABLE_BENCHMARKS, runner),
            lambda: _render_benchmarks(ctx, RENDERABLE_BENCHMARKS, runner),
            lambda: _rsi_splits(ctx, RSI_BENCHMARKS, runner),
        ):
            code = step()
            if code:
                return code
        return 0

    print(usage(), file=sys.stderr, end="")
    return _die(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
