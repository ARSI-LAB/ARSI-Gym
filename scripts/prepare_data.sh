#!/usr/bin/env bash
set -euo pipefail

# Compatibility shim for `agent-gym prepare-data`.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

export AGENT_GYM_REPO="${AGENT_GYM_REPO:-${repo_root}}"
export AGENT_GYM_OVERLAY="${AGENT_GYM_OVERLAY:-${repo_root}/overlay}"
export PYTHONPATH="${AGENT_GYM_OVERLAY}:${AGENT_GYM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    gym_bin="${GYM_BIN:-agent-gym}"
    gym_path="${gym_bin}"
    if [[ "${gym_path}" != */* ]]; then
        gym_path="$(command -v "${gym_bin}" 2>/dev/null || true)"
    fi
    if [[ -n "${gym_path}" && -x "$(dirname "${gym_path}")/python" ]]; then
        PYTHON_BIN="$(dirname "${gym_path}")/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

exec "${PYTHON_BIN}" -m agent_gym.prepare_data "$@"
