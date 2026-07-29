#include "support/test_utils.h"

#include "models/gwp05/pipeline.h"
#include "models/gwp05/state.h"

#include <string>

int main() {
    using wam::Backend;
    using wam::ComputePrecision;
    using wam::ErrorCode;
    using wam::internal::gwp05::ExecutionProfile;
    using wam::internal::gwp05::KernelDispatch;
    using wam::internal::gwp05::execution_profile_name;
    using wam::internal::gwp05::resolve_kernel_dispatch;
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
        ErrorCode::unsupported, "model must reject metadata-only backend");

    wam::internal::gwp05::ModelResources resources;
    resources.prompt_cache_limit = 2;
    auto first = wam::internal::gwp05::create_session_state(resources);
    auto second = wam::internal::gwp05::create_session_state(resources);
    require(first.get() != second.get(),
            "GWP Sessions must own distinct mutable state");
    const std::int32_t tokens[] = {3, 4};
    wam::internal::gwp05::PipelineInputsView input;
    input.lang_tokens = tokens;
    input.n_lang = 2;
    const std::vector<float> embedding = {1.0F, 2.0F};
    first->put_cached_prompt(input, embedding);
    std::vector<float> cached;
    require(first->get_cached_prompt(input, cached) && cached == embedding,
            "first GWP Session did not retain its prompt cache");
    require(!second->get_cached_prompt(input, cached),
            "GWP prompt cache leaked across Sessions");
    first->projected_prompt_signature = {9.0F};
    wam::internal::gwp05::reset_session(resources, *first);
    require(first->prompt_cache.empty() &&
                first->projected_prompt_signature.empty() &&
                first->prompt_cache_hits == 0 &&
                first->prompt_cache_misses == 0,
            "GWP reset did not clear Session caches and counters");
    return 0;
}
