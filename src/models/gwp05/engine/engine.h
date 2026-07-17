#pragma once

#include "models/gwp05/artifact.h"
#include "models/common/scheduler.h"
#include "models/gwp05/semantics.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace wam::internal::gwp05::engine {

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
    bool fused_image_preprocess = false;
    bool packed_qkv = false;
    bool cuda_graphs = false;
    bool single_token_timestep = false;
    bool unrolled_denoise = false;
};

KernelDispatch resolve_kernel_dispatch(Precision precision, Backend backend,
                                       const std::string & artifact_policy);
const char * execution_profile_name(ExecutionProfile profile) noexcept;

struct Config {
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
    float norm_eps = 0.0f;
    int num_steps = 0;
    int action_chunk = 0;
    float rms_eps = 0.0f;
    int rope_n_dims = 0;
    float rope_freq_base = 0.0f;
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
    float flow_shift = 0.0f;
};

enum class PixelFormat { U8, F32_RGB_01 };

struct ImageView {
    const void * data = nullptr;
    int w = 0;
    int h = 0;
    PixelFormat format = PixelFormat::U8;
};

struct EngineInputsView {
    const ImageView * images = nullptr;
    int n_images = 0;
    const float * precomputed_ref_latent = nullptr;
    int precomputed_ref_latent_n = 0;
    const float * precomputed_prompt_emb = nullptr;
    int precomputed_prompt_tokens = 0;
    const std::int32_t * lang_tokens = nullptr;
    int n_lang = 0;
    const float * state = nullptr;
    const float * noise = nullptr;
    const std::int32_t * attention_mask = nullptr;
    int attention_mask_n = 0;
    bool enable_prefix_cache = true;
};

struct EngineTelemetry {
    float ms_total = 0.0f;
    float ms_vision = 0.0f;
    float ms_inference = 0.0f;
    float ms_prefill = 0.0f;
    float ms_denoise = 0.0f;
    float ms_vae = 0.0f;
    float ms_vae_preprocess = 0.0f;
    float ms_vae_graph = 0.0f;
    float ms_umt5 = 0.0f;
    float ms_prefix_cache = 0.0f;
    float ms_prefix_graph_build = 0.0f;
    float ms_action_graph_build = 0.0f;
    float ms_prompt_projection = 0.0f;
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

struct EngineOptions {
    int device_index = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageEncoderPolicy language_encoder_policy = LanguageEncoderPolicy::resident;
    std::vector<std::int32_t> fixed_tokens;
    std::vector<std::int32_t> fixed_attention_mask;
    KernelDispatch dispatch{};
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual std::vector<float> predict(const EngineInputsView & inputs) = 0;
    virtual void reset() = 0;

    Config cfg{};
    EngineTelemetry stats{};
    std::vector<RuntimeComponentInfo> runtime_components;
    LanguageEncoderPolicy language_encoder_policy = LanguageEncoderPolicy::resident;
    bool text_encoder_resident = false;
    std::uint64_t resident_device_bytes = 0;
    std::uint64_t peak_component_device_bytes = 0;
};

inline int default_cpu_threads() {
    const unsigned count = std::thread::hardware_concurrency();
    return count == 0 ? 4 : static_cast<int>(std::min(count, 8u));
}

namespace gwp05_semantics {

inline constexpr std::int64_t kPromptTokens = semantics::kPromptTokens;
inline constexpr std::int32_t kVisualRopeSections[3] = {44, 42, 42};
using VisualPositions = semantics::VisualPositions;

inline std::vector<std::int32_t> action_positions(std::int64_t tokens, bool state) {
    return semantics::action_positions(tokens, state);
}
inline std::vector<float> action_timesteps(std::int64_t tokens, float timestep, bool state) {
    return semantics::action_timesteps(tokens, timestep, state);
}
inline VisualPositions visual_positions(std::int64_t height, std::int64_t width) {
    return semantics::visual_positions(height, width);
}
inline std::vector<float> flow_timesteps(int steps, float shift,
                                         std::vector<float> & sigmas) {
    FlowMatchEulerSchedule schedule = make_flow_match_euler_schedule(steps, shift);
    sigmas = std::move(schedule.sigmas);
    return std::move(schedule.timesteps);
}
inline bool validate_config(const Config & cfg, std::int64_t t5_capacity,
                            std::string & error) {
    try {
        (void) semantics::resolve_geometry({
            cfg.image_height, cfg.image_width, cfg.num_views, cfg.action_chunk,
            cfg.n_layers, cfg.num_steps, cfg.hidden, cfg.expert_h,
            cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, t5_capacity,
        });
    } catch (const std::exception & failure) {
        error = failure.what();
        return false;
    }
    return cfg.n_lang == kPromptTokens && cfg.n_state == 1 &&
        cfg.n_prefix == cfg.n_state + cfg.n_img &&
        cfg.n_full == cfg.n_prefix + cfg.n_suffix && cfg.n_suffix == cfg.action_chunk;
}

} // namespace gwp05_semantics

std::unique_ptr<Engine> create(const ArtifactContract & artifact,
                               const EngineOptions & options);

} // namespace wam::internal::gwp05::engine
