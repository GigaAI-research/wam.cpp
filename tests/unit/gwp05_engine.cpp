#include "support/test_utils.h"

#include "models/gwp05/engine/engine.h"

#include <string>

int main() {
    using wam::Backend;
    using wam::ComputePrecision;
    using wam::ErrorCode;
    using wam::internal::gwp05::engine::ExecutionProfile;
    using wam::internal::gwp05::engine::KernelDispatch;
    using wam::internal::gwp05::engine::execution_profile_name;
    using wam::internal::gwp05::engine::resolve_kernel_dispatch;
    using wam::test::require;
    using wam::test::require_error;

    const KernelDispatch f32 = resolve_kernel_dispatch(
        ComputePrecision::automatic, Backend::automatic, "legacy-source");
    require(f32.profile == ExecutionProfile::reference &&
                !f32.native_bf16 && !f32.unrolled_denoise,
            "automatic F32 dispatch changed");

    const KernelDispatch bf16 = resolve_kernel_dispatch(
        ComputePrecision::automatic, Backend::cuda,
        "mot-vae-bf16-qkv-v1");
    require(bf16.profile == ExecutionProfile::latency &&
                bf16.native_bf16 && bf16.bf16_vae && bf16.packed_qkv &&
                bf16.single_token_timestep && bf16.unrolled_denoise,
            "automatic native-BF16 dispatch changed");
    require(std::string(execution_profile_name(f32.profile)) == "reference" &&
                std::string(execution_profile_name(bf16.profile)) == "latency",
            "execution profile names changed");

    require_error(
        [] {
            (void) resolve_kernel_dispatch(
                ComputePrecision::bf16, Backend::automatic,
                "source-f32-v1");
        },
        ErrorCode::unsupported, "BF16 must reject an F32 artifact");
    require_error(
        [] {
            (void) resolve_kernel_dispatch(
                ComputePrecision::f16, Backend::automatic,
                "source-f32-v1");
        },
        ErrorCode::unsupported, "GWP must reject unsupported precision");
    require_error(
        [] {
            (void) resolve_kernel_dispatch(
                ComputePrecision::f32, Backend::cpu_metadata,
                "source-f32-v1");
        },
        ErrorCode::unsupported, "engine must reject metadata-only backend");
    return 0;
}
