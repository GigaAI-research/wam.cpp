#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_URL="${ROBOTWIN_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_REVISION="${ROBOTWIN_REVISION:-0aeea2d669c0f8516f4d5785f0aa33ba812c14b4}"
TARGET="${ROBOTWIN_TARGET:-${SCRIPT_DIR}/RoboTwin}"

if [[ -e "$TARGET" ]]; then
    echo "RoboTwin target already exists: $TARGET" >&2
    exit 1
fi

git clone "$ROBOTWIN_URL" "$TARGET"
git -C "$TARGET" checkout --detach "$ROBOTWIN_REVISION"
echo "RoboTwin checkout ready: $TARGET ($ROBOTWIN_REVISION)"
