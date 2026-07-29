#pragma once

#include "wam/runtime_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace wam::internal::gwp05 {

constexpr float kPi = 3.14159265358979323846F;

enum class LanguageExecutionMode {
    tokens,
    fixed_tokens,
    external_embedding,
};

enum class MotPrecisionPolicy {
    f32,
    native_bf16,
};

enum class ExecutionProfile {
    reference,
    latency,
};

struct KernelDispatch {
    ExecutionProfile profile = ExecutionProfile::reference;
    bool native_bf16 = false;
    bool bf16_output_gemm = false;
    bool f32_accumulation = true;
    bool bf16_hidden_and_kv = false;
    bool bf16_vae = false;
    bool packed_qkv = false;
    bool cuda_graphs = false;
    bool single_token_timestep = false;
    bool unrolled_denoise = false;
};

struct ModelGeometry {
    std::int64_t n_img = 0;
    std::int64_t n_lang = 0;
    std::int64_t n_state = 0;
    std::int64_t n_prefix = 0;
    std::int64_t n_suffix = 0;
    std::int64_t n_full = 0;
    std::int64_t hidden = 0;
    std::int64_t expert_h = 0;
    std::int64_t intermediate = 0;
    std::int64_t expert_inter = 0;
    std::int64_t n_q_heads = 0;
    std::int64_t n_kv_heads = 0;
    std::int64_t head_dim = 0;
    std::int64_t q_full_dim = 0;
    std::int64_t kv_full_dim = 0;
    std::int64_t n_layers = 0;
    std::int64_t max_state_dim = 0;
    std::int64_t max_action_dim = 0;
    std::int64_t real_state_dim = 0;
    std::int64_t real_action_dim = 0;
    float norm_eps = 0.0F;
    int num_steps = 0;
    int action_chunk = 0;
    float rms_eps = 0.0F;
    int rope_n_dims = 0;
    float rope_freq_base = 0.0F;
    std::int64_t image_height = 0;
    std::int64_t image_width = 0;
    std::int64_t num_views = 0;
    std::int64_t t5_hidden = 0;
    std::int64_t t5_ffn_dim = 0;
    std::int64_t t5_heads = 0;
    std::int64_t t5_head_dim = 0;
    std::int64_t t5_layers = 0;
    std::int64_t t5_vocab_size = 0;
    std::int64_t vae_z_dim = 0;
    std::int64_t num_embodiments = 0;
    std::int64_t embodiment_id = 0;
    float flow_shift = 0.0F;
};

struct PipelineInputsView {
    const float * composite_image = nullptr;
    int composite_image_n = 0;
    const float * precomputed_ref_latent = nullptr;
    int precomputed_ref_latent_n = 0;
    const float * precomputed_prompt_emb = nullptr;
    int precomputed_prompt_tokens = 0;
    const std::int32_t * lang_tokens = nullptr;
    int n_lang = 0;
    const float * model_state = nullptr;
    const float * action_noise = nullptr;
    const std::int32_t * attention_mask = nullptr;
    int attention_mask_n = 0;
    bool enable_prefix_cache = true;
};

struct PipelineTelemetry {
    float ms_total = 0.0F;
    float ms_vision = 0.0F;
    float ms_inference = 0.0F;
    float ms_prefill = 0.0F;
    float ms_denoise = 0.0F;
    float ms_vae = 0.0F;
    float ms_vae_preprocess = 0.0F;
    float ms_vae_graph = 0.0F;
    float ms_umt5 = 0.0F;
    float ms_prefix_cache = 0.0F;
    float ms_prefix_graph_build = 0.0F;
    float ms_action_graph_build = 0.0F;
    float ms_prompt_projection = 0.0F;
    std::vector<float> ms_denoise_steps;
    bool prompt_cache_hit = false;
    bool prompt_cache_enabled = false;
    std::uint64_t prompt_cache_hits = 0;
    std::uint64_t prompt_cache_misses = 0;
    std::size_t prompt_cache_entries = 0;
    std::size_t prompt_cache_bytes = 0;
    bool projected_prompt_cache_hit = false;
    std::uint64_t projected_prompt_hits = 0;
    std::uint64_t projected_prompt_misses = 0;
};

} // namespace wam::internal::gwp05
