#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python="${WAM_PYTHON_EXECUTABLE:-python3}"
cmake_bin="${WAM_CMAKE_EXECUTABLE:-cmake}"
protoc="${WAM_PROTOC_EXECUTABLE:-$(command -v protoc || true)}"
owned_work=0

if [[ -n "${1:-}" ]]; then
    work_root="$1"
    mkdir -p "${work_root}"
    if find "${work_root}" -mindepth 1 -print -quit | grep -q .; then
        printf 'release work directory must be empty: %s\n' "${work_root}" >&2
        exit 2
    fi
else
    work_root="$(mktemp -d -t wam-source-release.XXXXXX)"
    owned_work=1
fi
work_root="$(cd "${work_root}" && pwd -P)"

cleanup() {
    if [[ "${owned_work}" == 1 && "${WAM_KEEP_RELEASE_WORK:-0}" != 1 ]]; then
        rm -rf "${work_root}"
    else
        printf 'release work retained at %s\n' "${work_root}"
    fi
}
trap cleanup EXIT

[[ -n "${protoc}" ]] || {
    printf 'protoc is required to smoke-test the installed integration\n' >&2
    exit 2
}

archive="${work_root}/wam.cpp-0.1.0.tar.gz"
"${python}" "${source_root}/scripts/release/create_source_archive.py" \
    --source "${source_root}" --output "${archive}"
tar -xzf "${archive}" -C "${work_root}"

release_source="${work_root}/wam.cpp-0.1.0"
build_root="${work_root}/build"
install_root="${work_root}/install"
"${python}" "${release_source}/scripts/release/check_source_tree.py" "${release_source}"

configure=(
    "${cmake_bin}" -S "${release_source}" -B "${build_root}"
    -DCMAKE_BUILD_TYPE=Release
    -DWAM_CUDA=OFF
    -DWAM_BUILD_APPS=ON
    -DWAM_BUILD_SERVER=OFF
    -DWAM_BUILD_GIGA_WORLD_POLICY=ON
    "-DWAM_PROTOC_EXECUTABLE=${protoc}"
)
if [[ -n "${WAM_LLAMA_SOURCE_DIR:-}" ]]; then
    configure+=("-DWAM_LLAMA_SOURCE_DIR=${WAM_LLAMA_SOURCE_DIR}")
fi
if [[ -n "${WAM_LLAMA_ARCHIVE:-}" ]]; then
    configure+=("-DWAM_LLAMA_ARCHIVE=${WAM_LLAMA_ARCHIVE}")
fi
"${configure[@]}"
"${cmake_bin}" --build "${build_root}" --parallel "${WAM_BUILD_JOBS:-2}"
"${cmake_bin}" --install "${build_root}" --prefix "${install_root}"

integration_root="${install_root}/share/wam/integrations/giga-world-policy"
for installed in \
    "${install_root}/bin/run_inference_openloop_wam-cpp.sh" \
    "${install_root}/bin/wam_proto.py" \
    "${install_root}/bin/wam_rpc.py" \
    "${install_root}/share/wam/proto/wam.desc" \
    "${integration_root}/inference_openloop_wam_cpp.py" \
    "${integration_root}/upstream/0001-add-wam-cpp-entrypoints.patch" \
    "${integration_root}/upstream/scripts/run_inference_openloop_wam-cpp.sh" \
    "${integration_root}/upstream/scripts/inference_openloop_wam_cpp.py" \
    "${integration_root}/giga-world-policy.txt"; do
    [[ -f "${installed}" ]] || {
        printf 'installed integration file is missing: %s\n' "${installed}" >&2
        exit 1
    }
done

"${install_root}/bin/run_inference_openloop_wam-cpp.sh" --help >/dev/null
PYTHONPATH="${install_root}/bin" "${python}" -c \
    'from pathlib import Path; from wam_proto import load_types; import wam_rpc; load_types(Path(__import__("sys").argv[1]))' \
    "${install_root}/share/wam/proto/wam.desc"

printf 'clean source archive install smoke: pass\n'
