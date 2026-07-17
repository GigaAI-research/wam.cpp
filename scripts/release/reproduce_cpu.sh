#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${1:-${source_root}/build-release-cpu}"
install_root="${build_root}/install"

python3 "${source_root}/scripts/release/check_source_tree.py" "${source_root}"

configure=(
  cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release
  -DWAM_CUDA=OFF
  -DWAM_BUILD_APPS=ON
  -DWAM_BUILD_SERVER=OFF
)
if [[ -n "${WAM_LLAMA_SOURCE_DIR:-}" ]]; then
  configure+=("-DWAM_LLAMA_SOURCE_DIR=${WAM_LLAMA_SOURCE_DIR}")
fi
if [[ -n "${WAM_LLAMA_ARCHIVE:-}" ]]; then
  configure+=("-DWAM_LLAMA_ARCHIVE=${WAM_LLAMA_ARCHIVE}")
fi
if [[ -n "${WAM_PROTOC_EXECUTABLE:-}" ]]; then
  configure+=("-DWAM_PROTOC_EXECUTABLE=${WAM_PROTOC_EXECUTABLE}")
fi
"${configure[@]}"
cmake --build "${build_root}" --parallel "${WAM_BUILD_JOBS:-2}"
cmake --install "${build_root}" --prefix "${install_root}"
test -f "${install_root}/share/doc/wam/THIRD_PARTY_NOTICES.md"
test -x "${install_root}/bin/wam-cli"
"${install_root}/bin/wam-cli" --help >/dev/null
echo "clean CPU build and install: pass"
