#include "models/gwp05/model.h"

#include "arch.h"
#include "model_registry.h"
#include "models/gwp05/artifact.h"
#include "models/gwp05/engine/engine.h"
#include "models/gwp05/inputs.h"
#include "models/gwp05/semantics.h"
#include "wam/error.h"
#include "policy/action_decoder.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <utility>

namespace wam::internal::gwp05 {
namespace {

bool supports_tokens(policy::LanguageInputMode mode) {
    return mode == policy::LanguageInputMode::tokens ||
        mode == policy::LanguageInputMode::tokens_or_embedding;
}

bool supports_embedding(policy::LanguageInputMode mode) {
    return mode == policy::LanguageInputMode::embedding ||
        mode == policy::LanguageInputMode::tokens_or_embedding;
}

LanguageRuntimeMode resolve_language_mode(
    const RuntimeConfig & options, const policy::PolicySpec & spec) {
    LanguageRuntimeMode result = options.language_mode;
    if (result == LanguageRuntimeMode::automatic) {
        result = supports_tokens(spec.language.input_mode)
            ? LanguageRuntimeMode::tokens
            : LanguageRuntimeMode::external_embedding;
    }
    if (result == LanguageRuntimeMode::tokens &&
        !supports_tokens(spec.language.input_mode)) {
        throw Error(ErrorCode::unsupported,
                    "GWP artifact does not support token input",
                    {{"language_mode", "tokens"}});
    }
    if (result == LanguageRuntimeMode::external_embedding &&
        !supports_embedding(spec.language.input_mode)) {
        throw Error(ErrorCode::unsupported,
                    "GWP artifact does not support external embeddings",
                    {{"language_mode", "external_embedding"}});
    }
    if (options.fixed_prompt.has_value()) {
        if (result != LanguageRuntimeMode::tokens) {
            throw Error(ErrorCode::invalid_argument,
                        "fixed_prompt requires token language mode",
                        {{"fixed_prompt", "incompatible language mode"}});
        }
        if (options.fixed_prompt->token_ids.size() >
            spec.language.max_tokens) {
            throw Error(ErrorCode::invalid_argument,
                        "fixed_prompt exceeds PolicySpec capacity",
                        {{"fixed_prompt.token_ids", "too many tokens"}});
        }
    }
    return result;
}

Prediction make_prediction(const CoreAction & core,
                           const PreparedInputs & prepared,
                           const policy::PolicySpec & spec) {
    const policy::ActionSpec & action = spec.action;
    policy::validate_core_action(core.values, action);
    const policy::DecodedActionChunk action_chunk =
        policy::decode_action_reference(
            core.values, prepared.observation.raw_state, action, action.stats);
    Prediction prediction;
    prediction.action = policy::make_action_tensor(action_chunk);
    prediction.telemetry = core.stats;
    return prediction;
}

class Gwp05SessionImpl final : public SessionImpl {
public:
    Gwp05SessionImpl(std::shared_ptr<const ArtifactContract> artifact,
                     policy::PolicySpec policy_spec,
                     LanguageRuntimeMode language_mode,
                     std::optional<FixedPrompt> fixed_prompt,
                     engine::EngineSessionPtr engine_session,
                     std::uint64_t random_seed)
        : artifact_(std::move(artifact)),
          policy_spec_(std::move(policy_spec)),
          language_mode_(language_mode),
          fixed_prompt_(std::move(fixed_prompt)),
          engine_session_(std::move(engine_session)),
          random_seed_(random_seed),
          rng_(static_cast<std::mt19937::result_type>(random_seed)) {}

    Prediction predict(const Observation & inputs) override {
        using clock = std::chrono::steady_clock;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto total_begin = clock::now();
        const auto preprocess_begin = total_begin;
        PreparedInputs prepared = prepare_inputs(
            inputs, *artifact_, policy_spec_, language_mode_,
            rng_, fixed_prompt_);
        const double preprocess_milliseconds =
            std::chrono::duration<double, std::milli>(
                clock::now() - preprocess_begin).count();

        const CoreAction core = engine::predict(*engine_session_, prepared);
        const auto postprocess_begin = clock::now();
        Prediction prediction = make_prediction(core, prepared, policy_spec_);
        prediction.telemetry.preprocess_milliseconds = preprocess_milliseconds;
        prediction.telemetry.postprocess_milliseconds =
            std::chrono::duration<double, std::milli>(
                clock::now() - postprocess_begin).count();
        prediction.telemetry.total_milliseconds =
            std::chrono::duration<double, std::milli>(
                clock::now() - total_begin).count();
        return prediction;
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        engine::reset(*engine_session_);
        rng_.seed(static_cast<std::mt19937::result_type>(random_seed_));
    }

private:
    std::shared_ptr<const ArtifactContract> artifact_;
    policy::PolicySpec policy_spec_;
    LanguageRuntimeMode language_mode_ = LanguageRuntimeMode::automatic;
    std::optional<FixedPrompt> fixed_prompt_;
    engine::EngineSessionPtr engine_session_;
    std::uint64_t random_seed_ = 0;
    std::mt19937 rng_;
    std::mutex mutex_;
};

class Gwp05ModelImpl final : public ModelImpl {
public:
    Gwp05ModelImpl(ModelInfo info, policy::PolicySpec policy_spec,
                   std::shared_ptr<const ArtifactContract> artifact,
                   std::optional<FixedPrompt> fixed_prompt,
                   engine::EnginePtr engine)
        : ModelImpl(std::move(info), std::move(policy_spec)),
          artifact_(std::move(artifact)),
          fixed_prompt_(std::move(fixed_prompt)),
          engine_(std::move(engine)) {}

    std::unique_ptr<SessionImpl> create_session(
        const SessionConfig & options) override {
        if (!engine_) {
            throw Error(ErrorCode::unsupported,
                        "metadata-only GWP model cannot create a session",
                        {{"backend", "cpu_metadata"}});
        }
        engine::EngineSessionOptions engine_options;
        engine_options.enable_prefix_cache = options.enable_prefix_cache;
        return std::make_unique<Gwp05SessionImpl>(
            artifact_, policy_spec_, info_.language_mode, fixed_prompt_,
            engine::create_engine_session(*engine_, engine_options),
            options.random_seed);
    }

private:
    std::shared_ptr<const ArtifactContract> artifact_;
    std::optional<FixedPrompt> fixed_prompt_;
    engine::EnginePtr engine_;
};

void resolve_metadata_backend(ModelInfo & info) {
    info.backend = Backend::cpu_metadata;
    info.compute_precision = ComputePrecision::f32;
}

void set_capabilities(ModelInfo & info,
                      const policy::PolicySpec & spec,
                      const RuntimeConfig & options,
                      const ArtifactContract & artifact, bool action) {
    Capabilities & capabilities = info.capabilities;
    capabilities.action = action;
    capabilities.raw_images = true;
    capabilities.token_input =
        info.language_mode == LanguageRuntimeMode::tokens &&
        supports_tokens(spec.language.input_mode);
    capabilities.precomputed_embedding =
        info.language_mode == LanguageRuntimeMode::external_embedding &&
        supports_embedding(spec.language.input_mode);
    capabilities.explicit_action_noise = true;
    capabilities.batch_inference = false;
    capabilities.concurrent_sessions = false;
    capabilities.arbitrary_token_input =
        capabilities.token_input && !options.fixed_prompt.has_value();
    capabilities.fixed_token_input =
        capabilities.token_input && options.fixed_prompt.has_value();
    const bool native_bf16 =
        artifact.conversion_policy == "mot-bf16-v1" ||
        artifact.conversion_policy == "mot-vae-bf16-v1" ||
        artifact.conversion_policy == "mot-vae-bf16-qkv-v1";
    capabilities.backends = {Backend::cpu_metadata};
    if (!native_bf16) {
        capabilities.backends.push_back(Backend::automatic);
    }
#if WAM_HAS_CUDA
    if (native_bf16) {
        capabilities.backends.push_back(Backend::automatic);
    }
    capabilities.backends.push_back(Backend::cuda);
#endif
    capabilities.compute_precisions = {
        native_bf16 ? ComputePrecision::bf16 : ComputePrecision::f32};
}

} // namespace

std::unique_ptr<ModelImpl> create_model(
    const RuntimeConfig & options,
    ModelInfo info,
    std::optional<policy::PolicySpec> policy_spec,
    std::shared_ptr<GgufReader> reader) {
    if (reader == nullptr) {
        throw Error(ErrorCode::internal,
                    "GWP05 factory received a null GGUF reader");
    }
    policy::PolicySpec resolved_spec = policy_spec.has_value()
        ? std::move(*policy_spec)
        : read_legacy_policy_spec(*reader);
    std::shared_ptr<const ArtifactContract> artifact =
        load_artifact(std::move(reader), resolved_spec);

    info.language_mode = resolve_language_mode(options, resolved_spec);
    if (options.fixed_prompt.has_value()) {
        (void) semantics::prepare_prompt(
            options.fixed_prompt->token_ids,
            options.fixed_prompt->attention_mask,
            artifact->geometry.t5_vocab_size,
            resolved_spec.language.padding_side);
    }

    engine::EnginePtr runtime;
    if (options.backend == Backend::cpu_metadata) {
        if (options.compute_precision != ComputePrecision::automatic &&
            options.compute_precision != ComputePrecision::f32) {
            throw Error(ErrorCode::unsupported,
                        "metadata-only GWP loading accepts automatic or F32 precision");
        }
        resolve_metadata_backend(info);
        set_capabilities(info, resolved_spec, options, *artifact, false);
    } else {
        engine::EngineOptions engine_options;
        engine_options.backend = options.backend;
        engine_options.compute_precision = options.compute_precision;
        engine_options.device_index = options.device_index;
        engine_options.prompt_cache_capacity =
            options.prompt_cache_capacity;
        engine_options.language_mode = info.language_mode;
        engine_options.fixed_prompt = options.fixed_prompt;
        runtime = engine::create_engine(*artifact, engine_options);
        const engine::EngineInfo & runtime_info =
            engine::engine_info(*runtime);
        info.backend = runtime_info.backend;
        info.compute_precision = runtime_info.compute_precision;
        info.resident_device_bytes = runtime_info.resident_device_bytes;
        info.peak_component_device_bytes =
            runtime_info.peak_component_device_bytes;
        info.runtime_components = runtime_info.runtime_components;
        set_capabilities(info, resolved_spec, options, *artifact, true);
        info.capabilities.compute_precisions = {
            runtime_info.compute_precision};
    }

    return std::make_unique<Gwp05ModelImpl>(
        std::move(info), std::move(resolved_spec), std::move(artifact),
        options.fixed_prompt, std::move(runtime));
}

} // namespace wam::internal::gwp05

namespace wam::internal {

void register_gwp05(ModelRegistry & registry) {
    registry.add(Arch::gwp05, gwp05::create_model);
}

} // namespace wam::internal
