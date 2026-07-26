#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/LIBERO"
URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
REVISION="8f1084e3132a39270c3a13ebe37270a43ece2a01"

setup_libero() {
    if [[ -d "${TARGET}/.git" ]]; then
        git -C "${TARGET}" fetch origin "${REVISION}"
    elif [[ -e "${TARGET}" && -n "$(ls -A "${TARGET}")" ]]; then
        echo "Refusing to replace non-empty ${TARGET}" >&2
        return 1
    else
        rmdir "${TARGET}" 2>/dev/null || true
        git clone "${URL}" "${TARGET}"
    fi
    git -C "${TARGET}" checkout --detach "${REVISION}"
    echo "LIBERO checkout ready at ${TARGET} (${REVISION})"
}

setup_libero "$@"
