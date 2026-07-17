#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    cat <<'EOF'
Usage: run_inference_openloop_wam-cpp.sh server|client [arguments...]

Server environment:
  WAM_MODEL                 Converted GWP-0.5 GGUF (required)
  GIGA_WORLD_POLICY_ROOT    Frozen upstream checkout (required)
  WAM_LIBRARY               libwam_c_api.so (auto-discovered when installed)
  WAM_PROTO_DESCRIPTOR      wam.desc (auto-discovered when installed)
  FIXED_T5_PATH             Trusted fixed prompt embedding (.pt)
  WAM_FIXED_PROMPT_JSON     Fixed token JSON for WAM_LANGUAGE_POLICY=fixed

Client environment:
  GIGA_WORLD_POLICY_ROOT, CHECKPOINT, NORM_STATS, and DATA_PATH are required.
  DATA_IDX and REPLAN_STEPS remain optional and are passed to the upstream script.

Common optional environment:
  HOST=127.0.0.1 PORT=11444 DEVICE=0 WAM_INSTALL_PREFIX=<prefix>
EOF
}

die() {
    printf 'run_inference_openloop_wam-cpp.sh: %s\n' "$*" >&2
    exit 2
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "required environment variable ${name} is not set"
}

require_file() {
    local name="$1"
    local value="$2"
    [[ -f "${value}" ]] || die "${name} is not a file: ${value}"
}

discover_prefix() {
    if [[ -n "${WAM_INSTALL_PREFIX:-}" ]]; then
        printf '%s\n' "${WAM_INSTALL_PREFIX}"
        return
    fi
    if [[ "$(basename -- "${SCRIPT_DIR}")" == "bin" ]]; then
        dirname -- "${SCRIPT_DIR}"
        return
    fi
    if [[ -n "${WAM_LIBRARY:-}" ]]; then
        local library_dir
        library_dir="$(cd -- "$(dirname -- "${WAM_LIBRARY}")" 2>/dev/null && pwd -P)" || return 0
        case "$(basename -- "${library_dir}")" in
            lib|lib64)
                dirname -- "${library_dir}"
                ;;
        esac
    fi
}

first_file() {
    local candidate
    for candidate in "$@"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

resolve_adapter() {
    if [[ -n "${WAM_GWP_ADAPTER:-}" ]]; then
        printf '%s\n' "${WAM_GWP_ADAPTER}"
        return
    fi
    first_file \
        "${SCRIPT_DIR}/inference_openloop_wam_cpp.py" \
        "${WAM_PREFIX:-}/share/wam/integrations/giga-world-policy/inference_openloop_wam_cpp.py" \
        "${WAM_PREFIX:-}/libexec/wam/inference_openloop_wam_cpp.py" || true
}

resolve_library() {
    if [[ -n "${WAM_LIBRARY:-}" ]]; then
        printf '%s\n' "${WAM_LIBRARY}"
        return
    fi
    [[ -n "${WAM_PREFIX:-}" ]] || return 0
    first_file \
        "${WAM_PREFIX}/lib/libwam_c_api.so" \
        "${WAM_PREFIX}/lib64/libwam_c_api.so" \
        "${WAM_PREFIX}/lib/libwam_c_api.dylib" || true
}

resolve_descriptor() {
    if [[ -n "${WAM_PROTO_DESCRIPTOR:-}" ]]; then
        printf '%s\n' "${WAM_PROTO_DESCRIPTOR}"
        return
    fi
    [[ -n "${WAM_PREFIX:-}" ]] || return 0
    first_file "${WAM_PREFIX}/share/wam/proto/wam.desc" || true
}

parse_device() {
    local raw="${DEVICE:-0}"
    case "${raw}" in
        cuda)
            raw=0
            ;;
        cuda:*)
            raw="${raw#cuda:}"
            ;;
    esac
    [[ "${raw}" =~ ^[0-9]+$ ]] || die "DEVICE must be a non-negative integer or cuda:<index>, got: ${DEVICE:-0}"
    WAM_DEVICE_INDEX="$((10#${raw}))"

    if [[ "${WAM_BACKEND:-cuda}" == "cuda" && -v CUDA_VISIBLE_DEVICES ]]; then
        [[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]] || \
            die "WAM_BACKEND=cuda requires a non-empty CUDA_VISIBLE_DEVICES"
        local visible_count
        IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
        visible_count="${#visible_devices[@]}"
        (( WAM_DEVICE_INDEX < visible_count )) || die \
            "DEVICE=${DEVICE:-0} selects logical index ${WAM_DEVICE_INDEX}, but CUDA_VISIBLE_DEVICES exposes ${visible_count} device(s)"
    fi
}

run_server() {
    require_env WAM_MODEL
    require_env GIGA_WORLD_POLICY_ROOT
    require_file WAM_MODEL "${WAM_MODEL}"
    [[ -d "${GIGA_WORLD_POLICY_ROOT}" ]] || die \
        "GIGA_WORLD_POLICY_ROOT is not a directory: ${GIGA_WORLD_POLICY_ROOT}"

    WAM_PREFIX="$(discover_prefix)"
    WAM_LIBRARY="$(resolve_library)"
    WAM_PROTO_DESCRIPTOR="$(resolve_descriptor)"
    local adapter
    adapter="$(resolve_adapter)"

    [[ -n "${WAM_LIBRARY}" ]] || die \
        "WAM_LIBRARY is not set and libwam_c_api could not be found under WAM_INSTALL_PREFIX"
    [[ -n "${WAM_PROTO_DESCRIPTOR}" ]] || die \
        "WAM_PROTO_DESCRIPTOR is not set and wam.desc could not be found under WAM_INSTALL_PREFIX"
    [[ -n "${adapter}" ]] || die "GigaWorld-Policy adapter could not be found"
    require_file WAM_LIBRARY "${WAM_LIBRARY}"
    require_file WAM_PROTO_DESCRIPTOR "${WAM_PROTO_DESCRIPTOR}"
    require_file WAM_GWP_ADAPTER "${adapter}"

    local language_policy="${WAM_LANGUAGE_POLICY:-external_embedding}"
    local noise_policy="${WAM_NOISE_POLICY:-session}"
    case "${language_policy}" in
        resident|fixed|external_embedding) ;;
        *) die "unsupported WAM_LANGUAGE_POLICY: ${language_policy}" ;;
    esac
    case "${noise_policy}" in
        session|fixed|adapter-random) ;;
        *) die "unsupported WAM_NOISE_POLICY: ${noise_policy}" ;;
    esac
    if [[ "${language_policy}" == "fixed" ]]; then
        require_env WAM_FIXED_PROMPT_JSON
    elif [[ -n "${WAM_FIXED_PROMPT_JSON:-}" ]]; then
        die "WAM_FIXED_PROMPT_JSON requires WAM_LANGUAGE_POLICY=fixed"
    fi
    if [[ -n "${FIXED_T5_PATH:-}" ]]; then
        require_file FIXED_T5_PATH "${FIXED_T5_PATH}"
    fi
    if [[ -n "${WAM_FIXED_PROMPT_JSON:-}" ]]; then
        require_file WAM_FIXED_PROMPT_JSON "${WAM_FIXED_PROMPT_JSON}"
    fi
    if [[ "${noise_policy}" == "fixed" ]]; then
        require_env WAM_FIXED_NOISE
        require_file WAM_FIXED_NOISE "${WAM_FIXED_NOISE}"
    elif [[ -n "${WAM_FIXED_NOISE:-}" ]]; then
        die "WAM_FIXED_NOISE requires WAM_NOISE_POLICY=fixed"
    fi

    parse_device

    local python="${WAM_PYTHON_EXECUTABLE:-python3}"
    command -v "${python}" >/dev/null 2>&1 || die "Python executable was not found: ${python}"
    local serving_path="${WAM_SERVING_PYTHONPATH:-}"
    if [[ -z "${serving_path}" ]]; then
        if [[ -f "${SCRIPT_DIR}/../../scripts/serving/wam_proto.py" ]]; then
            serving_path="$(cd -- "${SCRIPT_DIR}/../../scripts/serving" && pwd -P)"
        elif [[ -n "${WAM_PREFIX}" && -f "${WAM_PREFIX}/bin/wam_proto.py" ]]; then
            serving_path="${WAM_PREFIX}/bin"
        else
            die "wam_proto.py could not be found; set WAM_SERVING_PYTHONPATH"
        fi
    fi

    local -a adapter_args=(
        --upstream-root "${GIGA_WORLD_POLICY_ROOT}"
        --library "${WAM_LIBRARY}"
        --model "${WAM_MODEL}"
        --descriptor "${WAM_PROTO_DESCRIPTOR}"
        --host "${HOST:-127.0.0.1}"
        --port "${PORT:-11444}"
        --backend "${WAM_BACKEND:-cuda}"
        --precision "${WAM_PRECISION:-bf16}"
        --device "${WAM_DEVICE_INDEX}"
        --prompt-cache-capacity "${WAM_PROMPT_CACHE_CAPACITY:-4}"
        --torch-cpu-threads "${WAM_TORCH_CPU_THREADS:-16}"
        --language-policy "${language_policy}"
        --noise-policy "${noise_policy}"
    )
    if [[ -n "${FIXED_T5_PATH:-}" ]]; then
        adapter_args+=(--fixed-t5 "${FIXED_T5_PATH}" --trust-fixed-t5)
    fi
    if [[ -n "${WAM_FIXED_PROMPT_JSON:-}" ]]; then
        adapter_args+=(--fixed-token-json "${WAM_FIXED_PROMPT_JSON}")
    fi
    if [[ -n "${WAM_FIXED_NOISE:-}" ]]; then
        adapter_args+=(--fixed-noise "${WAM_FIXED_NOISE}")
    fi
    if [[ -n "${WAM_RANDOM_SEED:-}" ]]; then
        adapter_args+=(--random-seed "${WAM_RANDOM_SEED}")
    fi
    if [[ "${WAM_ALLOW_UNSAFE_REMOTE:-0}" == "1" ]]; then
        adapter_args+=(--allow-unsafe-remote)
    fi
    if [[ "${WAM_DISABLE_PREFIX_CACHE:-0}" == "1" ]]; then
        adapter_args+=(--disable-prefix-cache)
    fi

    export PYTHONPATH="${serving_path}${PYTHONPATH:+:${PYTHONPATH}}"
    exec "${python}" "${adapter}" "${adapter_args[@]}" "$@"
}

run_client() {
    require_env GIGA_WORLD_POLICY_ROOT
    require_env CHECKPOINT
    require_env NORM_STATS
    require_env DATA_PATH
    local upstream_script="${GIGA_WORLD_POLICY_ROOT}/scripts/run_inference_openloop.sh"
    require_file UPSTREAM_RUN_INFERENCE "${upstream_script}"
    export HOST="${HOST:-127.0.0.1}"
    export PORT="${PORT:-11444}"
    cd -- "${GIGA_WORLD_POLICY_ROOT}/scripts"
    exec bash "${upstream_script}" client "$@"
}

case "${1:-}" in
    server)
        shift
        run_server "$@"
        ;;
    client)
        shift
        run_client "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    "")
        usage >&2
        exit 2
        ;;
    *)
        usage >&2
        die "unknown command: $1"
        ;;
esac
