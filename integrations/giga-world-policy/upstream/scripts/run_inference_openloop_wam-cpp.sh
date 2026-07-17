#!/usr/bin/env bash

# Thin upstream entry point for the installed wam.cpp GWP-0.5 adapter.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
UPSTREAM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

die() {
    printf 'run_inference_openloop_wam-cpp.sh: %s\n' "$*" >&2
    exit 2
}

resolve_runner() {
    if [[ -n "${WAM_CPP_RUNNER:-}" ]]; then
        printf '%s\n' "${WAM_CPP_RUNNER}"
        return
    fi
    if [[ -n "${WAM_INSTALL_PREFIX:-}" &&
          -x "${WAM_INSTALL_PREFIX}/bin/run_inference_openloop_wam-cpp.sh" ]]; then
        printf '%s\n' "${WAM_INSTALL_PREFIX}/bin/run_inference_openloop_wam-cpp.sh"
        return
    fi
    command -v run_inference_openloop_wam-cpp.sh || true
}

runner="$(resolve_runner)"
[[ -n "${runner}" ]] || die \
    "installed wam.cpp launcher not found; set WAM_INSTALL_PREFIX or WAM_CPP_RUNNER"
[[ -x "${runner}" ]] || die "wam.cpp launcher is not executable: ${runner}"

runner_real="$(cd -- "$(dirname -- "${runner}")" && pwd -P)/$(basename -- "${runner}")"
self_real="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
[[ "${runner_real}" != "${self_real}" ]] || die \
    "launcher resolved to this upstream shim; set WAM_INSTALL_PREFIX explicitly"

export GIGA_WORLD_POLICY_ROOT="${GIGA_WORLD_POLICY_ROOT:-${UPSTREAM_ROOT}}"
exec "${runner}" "$@"
