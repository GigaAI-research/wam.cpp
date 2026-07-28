#include "models/gwp05/engine/engine_internal.h"

#include "wam/error.h"
#include "runtime/telemetry.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace wam::internal::gwp05::engine {
namespace {

[[noreturn]] void unsupported(const std::string & message,
                              const std::string & field = {},
                              const std::string & reason = {}) {
    std::vector<ErrorDetail> details;
    if (!field.empty()) details.push_back({field, reason});
    throw Error(ErrorCode::unsupported, message, std::move(details));
}

LanguageExecutionMode resolve_language_mode(const EngineOptions & options) {
    if (options.fixed_prompt.has_value()) {
        if (options.language_mode != LanguageRuntimeMode::tokens) {
            throw Error(ErrorCode::failed_precondition,
                        "a fixed prompt requires token language mode");
        }
        return LanguageExecutionMode::fixed_tokens;
    }
    switch (options.language_mode) {
        case LanguageRuntimeMode::tokens:
            return LanguageExecutionMode::tokens;
        case LanguageRuntimeMode::external_embedding:
            return LanguageExecutionMode::external_embedding;
        case LanguageRuntimeMode::automatic:
            throw Error(ErrorCode::failed_precondition,
                        "GWP engine requires a resolved language mode");
    }
    throw Error(ErrorCode::failed_precondition,
                "invalid GWP language runtime mode");
}

std::vector<float> embedding_as_f32(const Tensor & embedding) {
    const std::size_t count =
        static_cast<std::size_t>(embedding.shape[0]) *
        static_cast<std::size_t>(embedding.shape[1]);
    std::vector<float> values(count);
    if (embedding.dtype == DType::f32) {
        std::memcpy(values.data(), embedding.data.data(),
                    count * sizeof(float));
        return values;
    }
    for (std::size_t index = 0; index < count; ++index) {
        std::uint16_t bf16 = 0;
        std::memcpy(&bf16,
                    embedding.data.data() + index * sizeof(bf16),
                    sizeof(bf16));
        const std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16U;
        std::memcpy(&values[index], &bits, sizeof(bits));
    }
    return values;
}

Telemetry public_stats(const Gwp05ModelArch & engine) {
    const EngineTelemetry & source = engine.stats;
    Telemetry output;
    output.model_milliseconds = source.ms_total;
    output.model_vision_milliseconds = source.ms_vision;
    output.model_text_milliseconds = source.ms_umt5;
    output.model_prefill_milliseconds =
        std::max(0.0, static_cast<double>(source.ms_prefill - source.ms_umt5));
    output.model_decode_milliseconds = source.ms_denoise;
    output.total_milliseconds = source.ms_total;
    output.peak_device_memory_bytes =
        std::max(engine.resident_device_bytes,
                 engine.peak_component_device_bytes);
    runtime::append_timing(output, "vision.preprocess", source.ms_vae_preprocess);
    runtime::append_timing(output, "vision.graph", source.ms_vae_graph);
    runtime::append_timing(output, "text.encoder", source.ms_umt5);
    runtime::append_timing(output, "prefill.prefix_cache", source.ms_prefix_cache);
    runtime::append_timing(output, "prefill.prefix_graph_build",
                  source.ms_prefix_graph_build);
    runtime::append_timing(output, "prefill.prompt_projection",
                  source.ms_prompt_projection);
    runtime::append_timing(output, "decode.action_graph_build",
                  source.ms_action_graph_build);
    for (std::size_t index = 0;
         index < source.ms_denoise_steps.size(); ++index) {
        char name[64];
        std::snprintf(name, sizeof(name), "decode.denoise_step_%02zu", index);
        runtime::append_timing(output, name, source.ms_denoise_steps[index]);
    }
    return output;
}

std::unique_ptr<EngineSessionState> make_session_state(
    const EngineResources & resources) {
    auto state = std::make_unique<EngineSessionState>();
    state->cfg = resources.cfg;
    state->runtime_components = resources.runtime_components;
    state->language_mode = resources.language_mode;
    state->text_encoder_resident = resources.text_encoder_resident;
    state->resident_device_bytes = resources.resident_device_bytes;
    state->peak_component_device_bytes =
        resources.peak_component_device_bytes;
    state->backend = resources.backend;
    state->backend_request = resources.backend_request;
    state->device_index = resources.device_index;
    state->component_loaded = resources.component_loaded;
    state->component_device_bytes = resources.component_device_bytes;
    state->component_load_milliseconds =
        resources.component_load_milliseconds;
    state->component_unload_milliseconds =
        resources.component_unload_milliseconds;
    state->load_state = resources.load_state;
    state->metadata_only = resources.metadata_only;
    state->precision_policy = resources.precision_policy;
    state->dispatch = resources.dispatch;
    state->tuning = resources.tuning;
    state->logger = resources.logger;
    state->debug_dumper = resources.debug_dumper;
    state->conversion_policy = resources.conversion_policy;
    state->n_threads = resources.n_threads;
    state->weights = resources.weights;
    state->prompt_cache_limit = resources.prompt_cache_limit;
    state->fixed_prompt = resources.fixed_prompt;
    state->t5_max_length = resources.t5_max_length;
    state->vae_latents_mean = resources.vae_latents_mean;
    state->vae_latents_std = resources.vae_latents_std;
    return state;
}

} // namespace

const char * execution_profile_name(ExecutionProfile profile) noexcept {
    return profile == ExecutionProfile::latency ? "latency" : "reference";
}

KernelDispatch resolve_kernel_dispatch(
    ComputePrecision precision, Backend backend,
    const std::string & artifact_policy) {
    if (backend == Backend::unknown || backend == Backend::cpu_metadata) {
        unsupported("GWP engine requires a compute backend", "backend",
                    "expected automatic or cuda");
    }
    if (precision == ComputePrecision::unknown) {
        unsupported("unknown GWP compute precision", "compute_precision",
                    "unknown");
    }

    const bool native_artifact =
        artifact_policy == "mot-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-qkv-v1";
    if (precision == ComputePrecision::automatic) {
        precision = native_artifact ? ComputePrecision::bf16
                                    : ComputePrecision::f32;
    }

    KernelDispatch result;
    if (precision == ComputePrecision::f32) {
        if (native_artifact) {
            unsupported("F32 execution requires an F32 GWP artifact",
                        "gwp05.conversion_policy", artifact_policy);
        }
        if (artifact_policy != "legacy-source" &&
            artifact_policy != "source-f32-v1") {
            unsupported("unsupported F32 GWP conversion policy",
                        "gwp05.conversion_policy", artifact_policy);
        }
        result.cuda_graphs = backend == Backend::cuda;
        return result;
    }
    if (precision != ComputePrecision::bf16) {
        unsupported("unsupported GWP execution precision",
                    "compute_precision", "expected f32 or bf16");
    }
    if (backend != Backend::automatic && backend != Backend::cuda) {
        unsupported("the BF16 latency profile requires CUDA", "backend",
                    "expected automatic or cuda");
    }
    if (!native_artifact) {
        unsupported("the BF16 latency profile requires a native BF16 artifact",
                    "gwp05.conversion_policy", artifact_policy);
    }

    result.profile = ExecutionProfile::latency;
    result.native_bf16 = true;
    result.bf16_output_gemm = true;
    result.bf16_hidden_and_kv = true;
    result.bf16_vae =
        artifact_policy == "mot-vae-bf16-v1" ||
        artifact_policy == "mot-vae-bf16-qkv-v1";
    result.packed_qkv = artifact_policy == "mot-vae-bf16-qkv-v1";
    result.cuda_graphs = true;
    result.single_token_timestep = true;
    result.unrolled_denoise = true;
    return result;
}

void Gwp05ModelArch::reset_core() {
    mot_graph.reset();
    unrolled_action_graph.reset();
    cached_action_graph.reset();
    prefix_graph.reset();
    prefix_storage.reset();
    prompt_projection_graph.reset();
    prompt_cache.clear();
    projected_prompt_signature.clear();
    prompt_cache_hits = 0;
    prompt_cache_misses = 0;
    projected_prompt_hits = 0;
    projected_prompt_misses = 0;
    stats = {};
}

EnginePtr create_engine(const ArtifactContract & artifact,
                        const EngineOptions & options) {
    if (!artifact.reader) {
        throw Error(ErrorCode::internal,
                    "GWP artifact has no backing reader");
    }
    auto model = std::make_unique<EngineResources>();
    model->logger = options.logger
        ? options.logger
        : std::make_shared<runtime::Logger>(RuntimeConfig{});
    model->debug_dumper = options.debug_dump
        ? options.debug_dump
        : std::make_shared<ggml_backend::DebugDump>();
    model->backend_request = options.backend;
    model->device_index = options.device_index;
    model->prompt_cache_limit = options.prompt_cache_capacity;
    model->tuning = options.tuning;
    model->language_mode = resolve_language_mode(options);
    model->dispatch = resolve_kernel_dispatch(
        options.compute_precision, options.backend,
        artifact.conversion_policy);
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
    cfg.max_state_dim = geometry.state_dim;
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
    cfg.norm_eps = geometry.norm_epsilon;
    cfg.rms_eps = geometry.norm_epsilon;
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
    cfg.rope_freq_base = 10000.0F;
    model->t5_max_length = geometry.t5_max_length;
    model->conversion_policy =
        artifact.conversion_policy == "legacy-source"
            ? "" : artifact.conversion_policy;
    model->precision_policy = model->dispatch.native_bf16
        ? MotPrecisionPolicy::NATIVE_BF16 : MotPrecisionPolicy::F32;
    model->vae_latents_mean =
        reader.optional_f32_array("gwp05.vae_latents_mean");
    model->vae_latents_std =
        reader.optional_f32_array("gwp05.vae_latents_std");

    if (model->vae_latents_mean.size() !=
            static_cast<std::size_t>(cfg.vae_z_dim) ||
        model->vae_latents_std.size() !=
            static_cast<std::size_t>(cfg.vae_z_dim)) {
        throw Error(ErrorCode::incompatible_artifact,
                    "GWP runtime metadata dimensions are inconsistent");
    }
    for (float value : model->vae_latents_std) {
        if (!(value > 0.0F) || !std::isfinite(value)) {
            throw Error(ErrorCode::incompatible_artifact,
                        "GWP VAE latent standard deviation is invalid");
        }
    }

    if (!init_backend(*model)) {
        throw Error(ErrorCode::resource_exhausted,
                    "failed to initialize the requested GWP backend");
    }
    switch (model->language_mode) {
        case LanguageExecutionMode::tokens:
            if (!load_resident_weights(reader, *model)) {
                throw Error(ErrorCode::resource_exhausted,
                            "failed to load resident GWP weights");
            }
            break;
        case LanguageExecutionMode::fixed_tokens: {
            if (!load_component(reader, *model, WeightComponent::t5)) {
                throw Error(ErrorCode::resource_exhausted,
                            "failed to load the GWP text encoder");
            }
            model->load_state = LoadState::text_only;
            const FixedPrompt & prompt = *options.fixed_prompt;
            EngineInputsView prompt_inputs;
            prompt_inputs.lang_tokens = prompt.token_ids.data();
            prompt_inputs.n_lang =
                static_cast<int>(prompt.token_ids.size());
            prompt_inputs.attention_mask =
                prompt.attention_mask.data();
            prompt_inputs.attention_mask_n =
                static_cast<int>(prompt.attention_mask.size());
            PromptCacheEntry fixed;
            fixed.tokens = prompt.token_ids;
            fixed.mask = prompt.attention_mask;
            fixed.embedding = run_t5(*model, prompt_inputs);
            if (fixed.embedding.empty()) {
                throw Error(ErrorCode::inference_failed,
                            "failed to encode the fixed GWP prompt");
            }
            model->fixed_prompt = std::move(fixed);
            unload_component(*model, WeightComponent::t5);
            if (!load_component(reader, *model, WeightComponent::mot) ||
                !load_component(reader, *model, WeightComponent::vae)) {
                throw Error(ErrorCode::resource_exhausted,
                            "failed to load GWP compute weights");
            }
            model->load_state = LoadState::compute_only;
            break;
        }
        case LanguageExecutionMode::external_embedding:
            if (!load_component(reader, *model, WeightComponent::mot) ||
                !load_component(reader, *model, WeightComponent::vae)) {
                throw Error(ErrorCode::resource_exhausted,
                            "failed to load GWP compute weights");
            }
            model->load_state = LoadState::compute_only;
            break;
    }

    logf(*model, LogLevel::info,
        "gwp05: profile=%s policy=%s layers=%lld hidden=%lld "
        "action_hidden=%lld ref_tokens=%lld chunk=%d steps=%d",
        execution_profile_name(model->dispatch.profile),
        artifact.conversion_policy.c_str(),
        static_cast<long long>(cfg.n_layers),
        static_cast<long long>(cfg.hidden),
        static_cast<long long>(cfg.expert_h),
        static_cast<long long>(cfg.n_img), cfg.action_chunk, cfg.num_steps);
    EngineInfo info;
    info.backend = options.backend;
    info.compute_precision = model->dispatch.profile ==
            ExecutionProfile::latency
        ? ComputePrecision::bf16
        : ComputePrecision::f32;
    info.resident_device_bytes = model->resident_device_bytes;
    info.peak_component_device_bytes =
        model->peak_component_device_bytes;
    info.runtime_components = model->runtime_components;
    return EnginePtr(new Engine(std::move(model), std::move(info)));
}

std::vector<float> Gwp05ModelArch::predict_core(const EngineInputsView & in) {
    if (metadata_only) {
        logf(*this, LogLevel::error,
             "gwp05: predict is unavailable in metadata-only mode");
        return {};
    }
    using clock = std::chrono::steady_clock;
    stats = {};
    const auto total_begin = clock::now();
    if (!in.model_state) {
        logf(*this, LogLevel::error, "gwp05: state is required");
        return {};
    }
    for (int64_t i = 0; i < cfg.max_state_dim; ++i) {
        if (!std::isfinite(in.model_state[i])) {
            logf(*this, LogLevel::error, "gwp05: state contains NaN/Inf");
            return {};
        }
    }

    std::vector<float> model_state(
        in.model_state, in.model_state + cfg.max_state_dim);
    debug_dump(*this, "normalized_state", model_state, {cfg.max_state_dim});

    const auto vision_begin = clock::now();
    const size_t latent_count = static_cast<size_t>(cfg.vae_z_dim) *
                                (cfg.image_height / 16) * (cfg.image_width / 16);
    std::vector<float> reference;
    if (in.precomputed_ref_latent) {
        if (in.precomputed_ref_latent_n != static_cast<int>(latent_count)) {
            logf(*this, LogLevel::error,
                         "gwp05: precomputed reference latent has %d values, expected %zu",
                         in.precomputed_ref_latent_n, latent_count);
            return {};
        }
        reference.assign(in.precomputed_ref_latent,
                         in.precomputed_ref_latent + latent_count);
    } else {
        reference = run_vae(*this, in);
    }
    if (reference.size() != latent_count) return {};
    debug_dump(*this, "vae_latent", reference,
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
            logf(*this, LogLevel::error,
                         "gwp05: invalid precomputed prompt token count %d",
                         in.precomputed_prompt_tokens);
            return {};
        }
        const size_t count = static_cast<size_t>(in.precomputed_prompt_tokens) * cfg.t5_hidden;
        std::copy(in.precomputed_prompt_emb, in.precomputed_prompt_emb + count, prompt.begin());
    } else if (language_mode == LanguageExecutionMode::fixed_tokens) {
        if (!fixed_prompt || !in.lang_tokens || in.n_lang < 1 ||
            fixed_prompt->tokens.size() != static_cast<size_t>(in.n_lang) ||
            fixed_prompt->mask != prompt_mask(in) ||
            !std::equal(fixed_prompt->tokens.begin(), fixed_prompt->tokens.end(),
                        in.lang_tokens)) {
            logf(*this, LogLevel::error,
                 "gwp05: token input does not match fixed prompt");
            return {};
        }
        prompt = fixed_prompt->embedding;
        prompt_cache_hit = true;
    } else if (language_mode == LanguageExecutionMode::external_embedding) {
        logf(*this, LogLevel::error,
             "gwp05: external_embedding policy requires an embedding");
        return {};
    } else {
        int valid_tokens = 0;
        if (!in.lang_tokens || in.n_lang < 1 || in.n_lang > cfg.n_lang ||
            !validate_prompt_mask(in, valid_tokens)) {
            logf(*this, LogLevel::error, "gwp05: invalid raw prompt input");
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
    debug_dump(*this, "t5_embedding", prompt, {cfg.n_lang, cfg.t5_hidden});
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
        !debug_dump_enabled(*this) &&
        tuning.prefix_cache;
    if (use_prefix_cache) {
        const auto prefix_begin = clock::now();
        if (!build_prefix_cache(*this, model_state, reference, prompt)) {
            logf(*this, LogLevel::error,
                 "gwp05: prefix cache build/compute failed");
            return {};
        }
        stats.ms_prefix_cache = std::chrono::duration<float, std::milli>(
            clock::now() - prefix_begin).count();
        stats.ms_prefill += stats.ms_prefix_cache;
    }

    std::vector<float> action(static_cast<size_t>(cfg.action_chunk) * cfg.max_action_dim);
    if (!in.action_noise) {
        logf(*this, LogLevel::error,
             "gwp05: prepared action noise is required");
        return {};
    }
    std::copy(in.action_noise, in.action_noise + action.size(), action.begin());
    for (float value : action) {
        if (!std::isfinite(value)) {
            logf(*this, LogLevel::error,
                 "gwp05: action noise contains NaN/Inf");
            return {};
        }
    }

    const auto denoise_begin = clock::now();
    const FlowMatchEulerSchedule schedule =
        make_flow_match_euler_schedule(cfg.num_steps, cfg.flow_shift);
    const std::vector<float> & timesteps = schedule.timesteps;
    const std::vector<float> & sigmas = schedule.sigmas;
    stats.ms_denoise_steps.reserve(static_cast<size_t>(cfg.num_steps));
    debug_dump(*this, "flow_timesteps", timesteps, {cfg.num_steps});
    debug_dump(*this, "flow_sigmas", sigmas, {cfg.num_steps + 1});
    const bool capture_steps = any_debug_dump_enabled(*this) ||
                               !tuning.graph_cache;
    const char * backend_name = ggml_backend_name(backend);
    const bool cuda_scheduler = backend_name && std::strstr(backend_name, "CUDA") != nullptr;
    const bool cpu_scheduler = !cuda_scheduler || tuning.force_cpu_scheduler;
    const bool use_unrolled_denoise = use_prefix_cache && !capture_steps && !cpu_scheduler &&
        dispatch.unrolled_denoise;
    if (use_unrolled_denoise) {
        const auto unrolled_begin = clock::now();
        if (!run_unrolled_action_denoise(*this, action, timesteps, sigmas)) {
            logf(*this, LogLevel::error,
                 "gwp05: unrolled denoise failed");
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
        debug_dump(*this, stage_name, action, {cfg.action_chunk, cfg.max_action_dim});
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
                  *this, model_state, action, reference, prompt,
                  timesteps[step], dt, step == 0 || cpu_scheduler, step == 0,
                  (capture_steps || cpu_scheduler) ? &prediction : nullptr,
                  (capture_steps && !cpu_scheduler) ? &action : nullptr);
        if (!step_ok) {
            logf(*this, LogLevel::error,
                 "gwp05: MoT failed at denoise step %d", step);
            return {};
        }
        std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_dt", step);
        debug_dump(*this, stage_name, {dt}, {1});
        if (cpu_scheduler) {
            for (size_t i = 0; i < action.size(); ++i) action[i] += dt * prediction[i];
        }
        if (capture_steps) {
            if (step == 0) debug_dump(*this, "velocity_step0", prediction,
                                      {cfg.action_chunk, cfg.max_action_dim});
            std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_velocity", step);
            debug_dump(*this, stage_name, prediction, {cfg.action_chunk, cfg.max_action_dim});
            std::snprintf(stage_name, sizeof(stage_name), "denoise_%02d_action_out", step);
            debug_dump(*this, stage_name, action, {cfg.action_chunk, cfg.max_action_dim});
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

    stats.ms_total = std::chrono::duration<float, std::milli>(
        clock::now() - total_begin).count();
    debug_dump(*this, "normalized_action", action,
               {cfg.action_chunk, cfg.max_action_dim});
    return action;
}

EngineSessionPtr create_engine_session(
    Engine & instance, const EngineSessionOptions & options) {
    if (!instance.resources_) {
        throw Error(ErrorCode::failed_precondition,
                    "GWP engine has no model resources");
    }
    return EngineSessionPtr(new EngineSession(
        instance,
        make_session_state(*instance.resources_),
        options.enable_prefix_cache));
}

CoreAction predict(EngineSession & session, const PreparedInputs & inputs) {
    if (session.owner_ == nullptr || session.state_ == nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "GWP engine session is not initialized");
    }
    Gwp05ModelArch & instance = *session.state_;
    std::lock_guard<std::mutex> lock(session.owner_->execution_mutex_);
    const std::size_t image_values =
        static_cast<std::size_t>(instance.cfg.image_width) *
        static_cast<std::size_t>(instance.cfg.image_height) * 3;
    const std::size_t noise_values =
        static_cast<std::size_t>(instance.cfg.action_chunk) *
        static_cast<std::size_t>(instance.cfg.max_action_dim);
    if (inputs.observation.composite_image.width !=
            static_cast<std::uint32_t>(instance.cfg.image_width) ||
        inputs.observation.composite_image.height !=
            static_cast<std::uint32_t>(instance.cfg.image_height) ||
        inputs.observation.composite_image.channels != 3 ||
        inputs.observation.composite_image.layout != policy::TensorLayout::chw ||
        inputs.observation.composite_image.pixels.size() != image_values ||
        inputs.observation.model_state.size() !=
            static_cast<std::size_t>(instance.cfg.max_state_dim) ||
        inputs.observation.action_noise.size() != noise_values) {
        throw Error(ErrorCode::invalid_argument,
                    "prepared GWP engine input contract is invalid");
    }
    std::vector<float> prompt_embedding;
    EngineInputsView view;
    view.composite_image = inputs.observation.composite_image.pixels.data();
    view.composite_image_n = static_cast<int>(
        inputs.observation.composite_image.pixels.size());
    view.model_state = inputs.observation.model_state.data();
    view.action_noise = inputs.observation.action_noise.data();
    view.enable_prefix_cache =
        inputs.enable_prefix_cache && session.enable_prefix_cache_;
    if (inputs.language_mode == LanguageRuntimeMode::external_embedding) {
        prompt_embedding = embedding_as_f32(inputs.embedding);
        view.precomputed_prompt_emb = prompt_embedding.data();
        view.precomputed_prompt_tokens =
            static_cast<int>(inputs.embedding.shape[0]);
        view.attention_mask =
            inputs.embedding_attention_mask.data();
        view.attention_mask_n =
            static_cast<int>(inputs.embedding_attention_mask.size());
    } else {
        view.lang_tokens = inputs.token_ids.data();
        view.n_lang = static_cast<int>(inputs.token_ids.size());
        view.attention_mask = inputs.attention_mask.data();
        view.attention_mask_n =
            static_cast<int>(inputs.attention_mask.size());
    }

    CoreAction output;
    output.values = instance.predict_core(view);
    if (output.values.empty()) {
        throw Error(ErrorCode::inference_failed,
                    "GWP engine inference failed");
    }
    output.stats = public_stats(instance);
    return output;
}

void reset(EngineSession & session) {
    if (session.owner_ == nullptr || session.state_ == nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "GWP engine session is not initialized");
    }
    std::lock_guard<std::mutex> lock(session.owner_->execution_mutex_);
    session.state_->reset_core();
}

const EngineInfo & engine_info(const Engine & instance) noexcept {
    return instance.info_;
}

void EngineDeleter::operator()(Engine * instance) const noexcept {
    delete instance;
}

void EngineSessionDeleter::operator()(EngineSession * session) const noexcept {
    delete session;
}

} // namespace wam::internal::gwp05::engine
