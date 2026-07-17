#!/usr/bin/env bash
set -euo pipefail

: "${WAM_CUDA_ARCH:?set WAM_CUDA_ARCH to 80 or 89}"

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${1:-${source_root}/build-cuda-release}"
install_root="${build_root}/install"
protoc="${WAM_PROTOC_EXECUTABLE:-$(command -v protoc || true)}"

case "${WAM_CUDA_ARCH}" in
  80|89) ;;
  *) echo "WAM_CUDA_ARCH must be 80 or 89" >&2; exit 2 ;;
esac

[[ -n "${protoc}" ]] || {
  echo "protoc is required to validate the installed open-loop launcher" >&2
  exit 2
}

if [[ -n "${WAM_LLAMA_ARCHIVE:-}" && -n "${LLAMA_ARCHIVE:-}" &&
      "${WAM_LLAMA_ARCHIVE}" != "${LLAMA_ARCHIVE}" ]]; then
  echo "WAM_LLAMA_ARCHIVE and LLAMA_ARCHIVE refer to different files" >&2
  exit 2
fi
llama_archive="${WAM_LLAMA_ARCHIVE:-${LLAMA_ARCHIVE:-}}"

configure=(
  cmake -S "${source_root}" -B "${build_root}"
  -DCMAKE_BUILD_TYPE=Release
  -DWAM_CUDA=ON -DWAM_CUDNN=ON
  "-DCMAKE_CUDA_ARCHITECTURES=${WAM_CUDA_ARCH}"
  -DWAM_BUILD_APPS=ON
  -DWAM_BUILD_GIGA_WORLD_POLICY=ON
  "-DWAM_PROTOC_EXECUTABLE=${protoc}"
)
if [[ -n "${WAM_LLAMA_SOURCE_DIR:-}" ]]; then
  configure+=("-DWAM_LLAMA_SOURCE_DIR=${WAM_LLAMA_SOURCE_DIR}")
fi
if [[ -n "${llama_archive}" ]]; then
  configure+=("-DWAM_LLAMA_ARCHIVE=${llama_archive}")
fi
"${configure[@]}"
cmake --build "${build_root}" --parallel "${WAM_BUILD_JOBS:-2}"
cmake --install "${build_root}" --prefix "${install_root}"
"${install_root}/bin/run_inference_openloop_wam-cpp.sh" --help >/dev/null
echo "CUDA runtime build and install: pass"
