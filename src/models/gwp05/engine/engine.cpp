#include "engine.h"
#include "models/common/gguf_reader.h"
#ifdef WAM_GWP05_CUDA_PREPROCESS
#include "models/gwp05/kernels/cuda_preprocess.h"
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wam::internal::gwp05::engine {

const char * execution_profile_name(ExecutionProfile profile) noexcept {
    return profile == ExecutionProfile::latency ? "latency" : "reference";
}

KernelDispatch resolve_kernel_dispatch(Precision precision, Backend backend,
                                       const std::string & artifact_policy) {
    KernelDispatch result;
    if (precision == Precision::f32_reference) {
        result.cuda_graphs = backend == Backend::cuda;
        return result;
    }
    if (precision != Precision::bf16_latency) {
        throw Error(ErrorCode::unsupported, "unsupported execution precision");
    }
    if (backend != Backend::cuda) {
        throw Error(ErrorCode::unsupported,
                    "the BF16 latency profile requires the CUDA backend");
    }
    const bool mot_bf16 = artifact_policy == "mot-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-qkv-v1";
    if (!mot_bf16) {
        throw Error(ErrorCode::unsupported,
                    "the BF16 latency profile requires a native BF16 artifact",
                    {{"gwp05.conversion_policy", artifact_policy}});
    }
    result.profile = ExecutionProfile::latency;
    result.native_bf16 = true;
    result.bf16_output_gemm = true;
    result.bf16_hidden_and_kv = true;
    result.bf16_vae = artifact_policy == "mot-vae-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-qkv-v1";
    result.fused_image_preprocess = result.bf16_vae;
    result.packed_qkv = artifact_policy == "mot-vae-bf16-qkv-v1";
    result.cuda_graphs = true;
    result.single_token_timestep = true;
    result.unrolled_denoise = true;
    return result;
}

namespace {
#include "weights.cpp"
#include "text_encoder.cpp"
#include "vae.cpp"
#include "mot.cpp"

} // namespace

void Gwp05ModelArch::reset() {
    mot_graph.reset();
    unrolled_action_graph.reset();
    cached_action_graph.reset();
    prefix_graph.reset();
    prefix_storage.reset();
    prompt_projection_graph.reset();
    projected_prompt_signature.clear();
}

std::unique_ptr<Engine> create(const ArtifactContract & artifact,
                               const EngineOptions & options) {
    auto model = std::make_unique<Gwp05ModelArch>();
    model->device_index = options.device_index;
    model->prompt_cache_limit = options.prompt_cache_capacity;
    model->dispatch = options.dispatch;
    model->language_encoder_policy = options.language_encoder_policy;
    if (!artifact.reader) return nullptr;
    GgufReader & reader = *artifact.reader;
    const Geometry & geometry = artifact.geometry;
    Config & cfg = model->cfg;
    cfg.hidden = geometry.hidden;
    cfg.n_layers = geometry.layers;
    cfg.n_q_heads = geometry.heads;
    cfg.n_kv_heads = geometry.heads;
    cfg.head_dim = geometry.head_dim;
    cfg.q_full_dim = cfg.n_q_heads * cfg.head_dim;
    cfg.kv_full_dim = cfg.q_full_dim;
    cfg.intermediate = geometry.ffn_dim;
    cfg.expert_h = geometry.action_hidden;
    cfg.expert_inter = geometry.action_ffn_dim;
    cfg.max_state_dim = geometry.action_dim;
    cfg.max_action_dim = geometry.action_dim;
    cfg.real_state_dim = geometry.real_state_dim;
    cfg.real_action_dim = geometry.real_action_dim;
    cfg.num_embodiments = geometry.num_embodiments;
    cfg.embodiment_id = geometry.embodiment_id;
    cfg.image_height = geometry.image_height;
    cfg.image_width = geometry.image_width;
    cfg.num_views = geometry.num_views;
    cfg.action_chunk = static_cast<int>(geometry.action_chunk);
    cfg.num_steps = static_cast<int>(geometry.inference_steps);
    cfg.flow_shift = geometry.flow_shift;
    cfg.norm_eps = geometry.norm_eps;
    cfg.rms_eps = geometry.norm_eps;
    cfg.t5_vocab_size = geometry.t5_vocab_size;
    cfg.t5_hidden = geometry.t5_hidden;
    cfg.t5_ffn_dim = geometry.t5_ffn_dim;
    cfg.t5_heads = geometry.t5_heads;
    cfg.t5_head_dim = geometry.t5_head_dim;
    cfg.t5_layers = geometry.t5_layers;
    cfg.vae_z_dim = geometry.vae_z_dim;
    cfg.n_img = artifact.sequence_geometry.visual_tokens;
    cfg.n_lang = semantics::kPromptTokens;
    cfg.n_state = artifact.sequence_geometry.state_tokens;
    cfg.n_prefix = artifact.sequence_geometry.prefix_tokens;
    cfg.n_suffix = artifact.sequence_geometry.action_tokens;
    cfg.n_full = artifact.sequence_geometry.full_tokens;
    cfg.rope_n_dims = static_cast<int>(cfg.head_dim);
    cfg.rope_freq_base = 10000.0f;
    model->t5_max_length = geometry.t5_max_length;
    model->conversion_policy = artifact.policy == "legacy-source" ? "" : artifact.policy;
    model->precision_policy = options.dispatch.native_bf16
        ? MotPrecisionPolicy::NATIVE_BF16 : MotPrecisionPolicy::F32;
    std::fprintf(stderr, "wam(gwp05): precision policy=%s gwp=%s vae=%s t5=BF16 stats=F32\n",
                 artifact.policy.c_str(), options.dispatch.native_bf16 ? "BF16" : "F32",
                 options.dispatch.bf16_vae ? "BF16" : "F32");
    model->state_q01 = reader.read_f32_tensor("state_q01");
    model->state_q99 = reader.read_f32_tensor("state_q99");
    model->action_q01 = reader.read_f32_tensor("action_q01");
    model->action_q99 = reader.read_f32_tensor("action_q99");
    model->vae_latents_mean = reader.optional_f32_array("gwp05.vae_latents_mean");
    model->vae_latents_std = reader.optional_f32_array("gwp05.vae_latents_std");
    if (model->state_q01.empty() || model->state_q99.empty() ||
        model->action_q01.empty() || model->action_q99.empty() ||
        model->vae_latents_mean.size() != static_cast<size_t>(model->cfg.vae_z_dim) ||
        model->vae_latents_std.size() != static_cast<size_t>(model->cfg.vae_z_dim)) {
        return nullptr;
    }
    for (float value : model->vae_latents_std) {
        if (!(value > 0.0f) || !std::isfinite(value)) {
            std::fprintf(stderr, "wam(gwp05): invalid VAE latent standard deviation\n");
            return nullptr;
        }
    }
    model->metadata_only = false;
    if (!model->metadata_only) {
        if (!init_backend(*model)) return nullptr;
        switch (options.language_encoder_policy) {
            case LanguageEncoderPolicy::resident:
                if (!load_resident_weights(reader, *model)) return nullptr;
                break;
            case LanguageEncoderPolicy::fixed: {
                if (!load_component(reader, *model, WeightComponent::t5)) return nullptr;
                model->load_state = LoadState::text_only;
                EngineInputsView prompt_inputs;
                prompt_inputs.lang_tokens = options.fixed_tokens.data();
                prompt_inputs.n_lang = static_cast<int>(options.fixed_tokens.size());
                prompt_inputs.attention_mask = options.fixed_attention_mask.data();
                prompt_inputs.attention_mask_n =
                    static_cast<int>(options.fixed_attention_mask.size());
                PromptCacheEntry fixed;
                fixed.tokens = options.fixed_tokens;
                fixed.mask = options.fixed_attention_mask;
                fixed.embedding = run_t5(*model, prompt_inputs);
                if (fixed.embedding.empty()) {
                    model->load_state = LoadState::failed;
                    return nullptr;
                }
                model->fixed_prompt = std::move(fixed);
                unload_component(*model, WeightComponent::t5);
                if (!load_component(reader, *model, WeightComponent::mot) ||
                    !load_component(reader, *model, WeightComponent::vae)) {
                    model->load_state = LoadState::failed;
                    return nullptr;
                }
                model->load_state = LoadState::compute_only;
                break;
            }
            case LanguageEncoderPolicy::external_embedding:
                if (!load_component(reader, *model, WeightComponent::mot) ||
                    !load_component(reader, *model, WeightComponent::vae)) {
                    model->load_state = LoadState::failed;
                    return nullptr;
                }
                model->load_state = LoadState::compute_only;
                break;
            default:
                return nullptr;
        }
    }
    std::fprintf(stderr, "wam(gwp05): layers=%lld hidden=%lld action_hidden=%lld ref_tokens=%lld "
                "chunk=%d steps=%d\n",
                static_cast<long long>(model->cfg.n_layers),
                static_cast<long long>(model->cfg.hidden),
                static_cast<long long>(model->cfg.expert_h),
                static_cast<long long>(model->cfg.n_img), model->cfg.action_chunk,
                model->cfg.num_steps);
    return model;
}

std::vector<float> Gwp05ModelArch::predict(const EngineInputsView & in) {
    if (metadata_only) {
        std::fprintf(stderr, "wam(gwp05): predict is unavailable in metadata-only mode\n");
        return {};
    }
    using clock = std::chrono::steady_clock;
    stats = {};
    const auto total_begin = clock::now();
    if (!in.state) {
        std::fprintf(stderr, "wam(gwp05): state is required\n");
        return {};
    }
    for (int64_t i = 0; i < cfg.real_state_dim; ++i) {
        if (!std::isfinite(in.state[i])) {
            std::fprintf(stderr, "wam(gwp05): state contains NaN/Inf\n");
            return {};
        }
    }

    std::vector<float> normalized_state(cfg.max_state_dim, 0.0f);
    for (int64_t i = 0; i < cfg.real_state_dim; ++i) {
        const float range = std::max(state_q99[i] - state_q01[i], 1e-8f);
        normalized_state[i] = ((in.state[i] - state_q01[i]) / range) * 2.0f - 1.0f;
    }
    debug_dump("normalized_state", normalized_state, {cfg.max_state_dim});

    const auto vision_begin = clock::now();
    const size_t latent_count = static_cast<size_t>(cfg.vae_z_dim) *
                                (cfg.image_height / 16) * (cfg.image_width / 16);
    std::vector<float> reference;
    if (in.precomputed_ref_latent) {
        if (in.precomputed_ref_latent_n != static_cast<int>(latent_count)) {
            std::fprintf(stderr, "wam(gwp05): precomputed reference latent has %d values, expected %zu\n",
                         in.precomputed_ref_latent_n, latent_count);
            return {};
        }
        reference.assign(in.precomputed_ref_latent,
                         in.precomputed_ref_latent + latent_count);
    } else {
        reference = run_vae(*this, in);
    }
    if (reference.size() != latent_count) return {};
    debug_dump("vae_latent", reference,
               {cfg.vae_z_dim, cfg.image_height / 16, cfg.image_width / 16});
    for (float value : reference) if (!std::isfinite(value)) return {};
    stats.ms_vision = std::chrono::duration<float, std::milli>(clock::now() - vision_begin).count();
    stats.ms_vae = stats.ms_vision;

    const auto prompt_begin = clock::now();
    std::vector<float> prompt(static_cast<size_t>(cfg.n_lang) * cfg.t5_hidden, 0.0f);
    bool prompt_cache_hit = false;
    bool ran_t5 = false;
    if (in.precomputed_prompt_emb) {
        if (in.precomputed_prompt_tokens < 1 || in.precomputed_prompt_tokens > cfg.n_lang) {
            std::fprintf(stderr, "wam(gwp05): invalid precomputed prompt token count %d\n",
                         in.precomputed_prompt_tokens);
            return {};
        }
        const size_t count = static_cast<size_t>(in.precomputed_prompt_tokens) * cfg.t5_hidden;
        std::copy(in.precomputed_prompt_emb, in.precomputed_prompt_emb + count, prompt.begin());
    } else if (language_encoder_policy == LanguageEncoderPolicy::fixed) {
        if (!fixed_prompt || !in.lang_tokens || in.n_lang < 1 ||
            fixed_prompt->tokens.size() != static_cast<size_t>(in.n_lang) ||
            fixed_prompt->mask != prompt_mask(in) ||
            !std::equal(fixed_prompt->tokens.begin(), fixed_prompt->tokens.end(),
                        in.lang_tokens)) {
            std::fprintf(stderr, "wam(gwp05): token input does not match fixed prompt\n");
            return {};
        }
        prompt = fixed_prompt->embedding;
        prompt_cache_hit = true;
    } else if (language_encoder_policy == LanguageEncoderPolicy::external_embedding) {
        std::fprintf(stderr, "wam(gwp05): external_embedding policy requires an embedding\n");
        return {};
    } else {
        int valid_tokens = 0;
        if (!in.lang_tokens || in.n_lang < 1 || in.n_lang > cfg.n_lang ||
            !validate_prompt_mask(in, valid_tokens)) {
            std::fprintf(stderr, "wam(gwp05): invalid raw prompt input\n");
            return {};
        }
        if ((prompt_cache_hit = get_cached_prompt(in, prompt))) {
        // The cached embedding already includes the zero-padded language suffix.
        } else {
            prompt = run_t5(*this, in);
            ran_t5 = true;
        }
    }
    if (prompt.size() != static_cast<size_t>(cfg.n_lang) * cfg.t5_hidden) return {};
    debug_dump("t5_embedding", prompt, {cfg.n_lang, cfg.t5_hidden});
    for (float value : prompt) if (!std::isfinite(value)) return {};
    if (!in.precomputed_prompt_emb && !prompt_cache_hit) {
        put_cached_prompt(in, prompt);
    }
    stats.ms_prefill = std::chrono::duration<float, std::milli>(clock::now() - prompt_begin).count();
    stats.ms_umt5 = ran_t5 ? stats.ms_prefill : 0.0f;
    stats.prompt_cache_hit = prompt_cache_hit;
    stats.prompt_cache_enabled = prompt_cache_limit != 0;
    stats.prompt_cache_hits = prompt_cache_hits;
    stats.prompt_cache_misses = prompt_cache_misses;
    stats.prompt_cache_entries = prompt_cache.size();
    stats.prompt_cache_bytes = prompt_cache_embedding_bytes();

    const char * runtime_backend_name = ggml_backend_name(backend);
    const bool use_prefix_cache = in.enable_prefix_cache && runtime_backend_name &&
        std::strstr(runtime_backend_name, "CUDA") != nullptr &&
        !debug_dump_enabled() &&
        std::getenv("WAM_GWP05_DISABLE_PREFIX_CACHE") == nullptr;
    if (use_prefix_cache) {
        const auto prefix_begin = clock::now();
        if (!build_prefix_cache(*this, normalized_state, reference, prompt)) {
            std::fprintf(stderr, "wam(gwp05): prefix cache build/compute failed\n");
            return {};
        }
        stats.ms_prefix_cache = std::chrono::duration<float, std::milli>(
            clock::now() - prefix_begin).count();
        stats.ms_prefill += stats.ms_prefix_cache;
    }

    std::vector<float> action(static_cast<size_t>(cfg.action_chunk) * cfg.max_action_dim);
    if (in.noise) {
        std::copy(in.noise, in.noise + action.size(), action.begin());
        for (float value : action) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "wam(gwp05): action noise contains NaN/Inf\n");
                return {};
            }
        }
    } else {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float & value : action) value = normal(rng);
    }

    const auto denoise_begin = clock::now();
    std::vector<float> sigmas;
    const std::vector<float> timesteps =
        gwp05_semantics::flow_timesteps(cfg.num_steps, cfg.flow_shift, sigmas);
    stats.ms_denoise_steps.reserve(static_cast<size_t>(cfg.num_steps));
    debug_dump("flow_timesteps", timesteps, {cfg.num_steps});
    debug_dump("flow_sigmas", sigmas, {cfg.num_steps + 1});
    const bool capture_steps = any_debug_dump_enabled() ||
                               std::getenv("WAM_GWP05_DISABLE_MOT_GRAPH_CACHE") != nullptr;
    const char * backend_name = ggml_backend_name(backend);
    const bool cuda_scheduler = backend_name && std::strstr(backend_name, "CUDA") != nullptr;
    const bool cpu_scheduler = !cuda_scheduler ||
                               std::getenv("WAM_GWP05_CPU_SCHEDULER") != nullptr;
    const bool use_unrolled_denoise = use_prefix_cache && !capture_steps && !cpu_scheduler &&
        dispatch.unrolled_denoise;
    if (use_unrolled_denoise) {
        const auto unrolled_begin = clock::now();
        if (!run_unrolled_action_denoise(*this, action, timesteps, sigmas)) {
            std::fprintf(stderr, "wam(gwp05): unrolled denoise failed\n");
            return {};
        }
        const float unrolled_ms = std::chrono::duration<float, std::milli>(
            clock::now() - unrolled_begin).count();
        for (int step = 0; step < cfg.num_steps; ++step) {
            stats.ms_denoise_steps.push_back(unrolled_ms / cfg.num_steps);
        }
    } else for (int step = 0; step < cfg.num_steps; ++step) {
        const auto step_begin = clock::now();
        char stage_name[64];
        std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_action_in", step);
        debug_dump(stage_name, action, {cfg.action_chunk, cfg.max_action_dim});
        const float dt = sigmas[step + 1] - sigmas[step];
        std::vector<float> prediction;
        const bool step_ok = use_prefix_cache
            ? run_cached_action_step(
                  *this, action, prompt, timesteps[step], dt,
                  step == 0 || cpu_scheduler,
                  (capture_steps || cpu_scheduler) ? &prediction : nullptr,
                  (capture_steps && !cpu_scheduler) ? &action : nullptr,
                  step)
            : run_mot_step(
                  *this, normalized_state, action, reference, prompt,
                  timesteps[step], dt, step == 0 || cpu_scheduler, step == 0,
                  (capture_steps || cpu_scheduler) ? &prediction : nullptr,
                  (capture_steps && !cpu_scheduler) ? &action : nullptr);
        if (!step_ok) {
            std::fprintf(stderr, "wam(gwp05): MoT failed at denoise step %d\n", step);
            return {};
        }
        std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_dt", step);
        debug_dump(stage_name, {dt}, {1});
        if (cpu_scheduler) {
            for (size_t i = 0; i < action.size(); ++i) action[i] += dt * prediction[i];
        }
        if (capture_steps) {
            if (step == 0) debug_dump("velocity_step0", prediction,
                                      {cfg.action_chunk, cfg.max_action_dim});
            std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_velocity", step);
            debug_dump(stage_name, prediction, {cfg.action_chunk, cfg.max_action_dim});
            std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_action_out", step);
            debug_dump(stage_name, action, {cfg.action_chunk, cfg.max_action_dim});
        }
        stats.ms_denoise_steps.push_back(
            std::chrono::duration<float, std::milli>(clock::now() - step_begin).count());
    }
    if (!use_unrolled_denoise && !capture_steps && !cpu_scheduler) {
        ggml_tensor * device_action = use_prefix_cache
            ? cached_action_graph->action_input : mot_graph->action_input;
        get_f32_tensor(device_action, action);
    }
    stats.ms_denoise = std::chrono::duration<float, std::milli>(clock::now() - denoise_begin).count();
    stats.ms_inference = stats.ms_prefill + stats.ms_denoise;

    std::vector<float> result(static_cast<size_t>(cfg.action_chunk) * cfg.real_action_dim);
    for (int step = 0; step < cfg.action_chunk; ++step) {
        for (int64_t dim = 0; dim < cfg.real_action_dim; ++dim) {
            const float range = std::max(action_q99[dim] - action_q01[dim], 1e-8f);
            float value = ((action[static_cast<size_t>(step) * cfg.max_action_dim + dim] + 1.0f) *
                           0.5f) * range + action_q01[dim];
            if ((dim >= 0 && dim < 6) || (dim >= 7 && dim < 13)) value += in.state[dim];
            result[static_cast<size_t>(step) * cfg.real_action_dim + dim] = value;
        }
    }
    stats.ms_total = std::chrono::duration<float, std::milli>(clock::now() - total_begin).count();
    debug_dump("action", result, {cfg.action_chunk, cfg.real_action_dim});
    return result;
}

} // namespace wam
