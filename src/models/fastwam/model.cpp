#include "models/fastwam/model.h"

#include "backends/ggml/debug_dump.h"
#include "runtime/logger.h"

#include "models/fastwam/contract.h"
#include "models/fastwam/inputs.h"
#include "models/fastwam/pipeline.h"
#include "models/fastwam/state.h"
#include "model_internal.h"
#include "wam/error.h"
#include "policy/action_decoder.h"

#include <chrono>
#include <mutex>
#include <random>
#include <utility>

namespace wam::internal::fastwam {
namespace {

Prediction make_prediction(const CoreAction & core,
                           const PreparedInputs & inputs,
                           const policy::PolicySpec & policy_spec) {
    policy::validate_core_action(core.values, policy_spec.action);
    const policy::DecodedActionChunk action = policy::decode_action_reference(
        core.values, inputs.observation.raw_state, policy_spec.action,
        policy_spec.action.stats);
    Prediction prediction;
    prediction.action = policy::make_action_tensor(action);
    prediction.telemetry = core.stats;
    return prediction;
}

class FastWamSessionImpl final : public SessionImpl {
public:
    FastWamSessionImpl(std::shared_ptr<const FastWamContract> contract,
                       policy::PolicySpec policy_spec,
                       std::shared_ptr<ModelResources> resources,
                       std::unique_ptr<SessionState> state,
                       std::uint64_t random_seed)
        : contract_(std::move(contract)),
          policy_spec_(std::move(policy_spec)),
          resources_(std::move(resources)),
          state_(std::move(state)),
          random_seed_(random_seed),
          rng_(static_cast<std::mt19937::result_type>(random_seed)) {}

    Prediction predict(const Observation & inputs) override {
        using clock = std::chrono::steady_clock;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto total_begin = clock::now();
        const auto preprocess_begin = total_begin;
        PreparedInputs prepared = prepare_inputs(
            inputs, *contract_, policy_spec_,
            LanguageRuntimeMode::external_embedding, rng_);
        const double preprocess_milliseconds =
            std::chrono::duration<double, std::milli>(
                clock::now() - preprocess_begin).count();

        const CoreAction core = run_pipeline(
            *resources_, *state_, *contract_, prepared);
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
        reset_session(*resources_, *state_);
        rng_.seed(static_cast<std::mt19937::result_type>(random_seed_));
    }

private:
    std::shared_ptr<const FastWamContract> contract_;
    policy::PolicySpec policy_spec_;
    std::shared_ptr<ModelResources> resources_;
    std::unique_ptr<SessionState> state_;
    std::uint64_t random_seed_ = 0;
    std::mt19937 rng_;
    std::mutex mutex_;
};

class FastWamModelImpl final : public ModelImpl {
public:
    FastWamModelImpl(ModelInfo info, policy::PolicySpec policy_spec,
                     std::shared_ptr<const FastWamContract> contract,
                     std::shared_ptr<ModelResources> resources)
        : ModelImpl(std::move(info), std::move(policy_spec)),
          contract_(std::move(contract)), resources_(std::move(resources)) {}

    std::unique_ptr<SessionImpl> create_session(
        const SessionConfig & options) override {
        if (!resources_) {
            throw Error(ErrorCode::unsupported,
                        "metadata-only FastWAM model cannot create a session",
                        {{"backend", "cpu_metadata"}});
        }
        return std::make_unique<FastWamSessionImpl>(
            contract_, policy_spec_, resources_, create_session_state(),
            options.random_seed);
    }

private:
    std::shared_ptr<const FastWamContract> contract_;
    std::shared_ptr<ModelResources> resources_;
};

} // namespace

std::unique_ptr<ModelImpl> create_model(
    const RuntimeConfig & options,
    ModelInfo info,
    std::optional<policy::PolicySpec> policy_spec,
    std::shared_ptr<GgufReader> reader) {
    if (reader == nullptr || !policy_spec.has_value()) {
        throw Error(ErrorCode::incompatible_artifact,
                    "FastWAM requires a PolicySpec GGUF");
    }
    if (options.language_mode != LanguageRuntimeMode::automatic &&
        options.language_mode != LanguageRuntimeMode::external_embedding) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM Gate B requires external embedding language mode");
    }
    if (options.fixed_prompt.has_value()) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM external embedding mode does not accept fixed tokens");
    }
    policy::PolicySpec resolved_spec = std::move(*policy_spec);
    std::shared_ptr<const FastWamContract> contract =
        load_contract(std::move(reader), resolved_spec);
    info.language_mode = LanguageRuntimeMode::external_embedding;
    info.artifact_components = contract->components;
    std::shared_ptr<ModelResources> resources;
    if (options.backend == Backend::cpu_metadata) {
        if (options.compute_precision != ComputePrecision::automatic &&
            options.compute_precision != ComputePrecision::f32) {
            throw Error(ErrorCode::unsupported,
                        "metadata-only FastWAM loading accepts automatic or F32 precision");
        }
        info.backend = Backend::cpu_metadata;
        info.compute_precision = ComputePrecision::f32;
    } else {
        ModelOptions model_options;
        model_options.backend = options.backend;
        model_options.compute_precision = options.compute_precision;
        model_options.device_index = options.device_index;
        model_options.logger =
            std::make_shared<runtime::Logger>(options);
        model_options.debug_dump =
            std::make_shared<ggml_backend::DebugDump>(
                options.debug_dump,
                [logger = model_options.logger](std::string_view message) {
                    logger->log(LogLevel::warning, message);
                });
        LoadedModel loaded = load_model_resources(*contract, model_options);
        resources = std::move(loaded.resources);
        const runtime::EngineInfo & runtime_info = loaded.info;
        info.backend = runtime_info.backend;
        info.compute_precision = runtime_info.compute_precision;
        info.resident_device_bytes = runtime_info.resident_device_bytes;
        info.peak_component_device_bytes =
            runtime_info.peak_component_device_bytes;
        info.runtime_components = runtime_info.runtime_components;
    }
    info.capabilities.action = resources != nullptr;
    info.capabilities.raw_images = true;
    info.capabilities.precomputed_embedding = true;
    info.capabilities.explicit_action_noise = true;
    info.capabilities.concurrent_sessions = false;
    info.capabilities.backends = {Backend::cpu_metadata};
#if WAM_HAS_CUDA
    info.capabilities.backends.push_back(Backend::automatic);
    info.capabilities.backends.push_back(Backend::cuda);
#endif
    info.capabilities.compute_precisions = {ComputePrecision::bf16};
    return std::make_unique<FastWamModelImpl>(
        std::move(info), std::move(resolved_spec), std::move(contract),
        std::move(resources));
}

} // namespace wam::internal::fastwam
