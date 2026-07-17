#include "model.h"

#include "model_registry.h"
#include "models/common/input_validation.h"
#include "inputs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>

namespace wam::internal::gwp05 {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

class Gwp05Session final : public SessionImpl {
public:
    Gwp05Session(std::shared_ptr<Runtime> runtime,
                 std::shared_ptr<const ArtifactContract> artifact,
                 SessionOptions options, std::uint64_t id,
                 LanguageEncoderPolicy language_policy,
                 std::optional<FixedPrompt> fixed_prompt)
        : runtime_(std::move(runtime)), artifact_(std::move(artifact)),
          options_(options), id_(id), language_policy_(language_policy),
          fixed_prompt_(std::move(fixed_prompt)), random_(0) {}

    Prediction predict(const Inputs & inputs) override {
        if (!runtime_->engine) {
            throw Error(ErrorCode::unsupported,
                        "GWP prediction requires an initialized F32 compute backend");
        }
        if (language_policy_ == LanguageEncoderPolicy::external_embedding &&
            std::holds_alternative<TokenInput>(inputs.language)) {
            throw Error(ErrorCode::failed_precondition,
                        "external_embedding policy requires a precomputed embedding");
        }
        if (language_policy_ == LanguageEncoderPolicy::fixed) {
            if (const auto * tokens = std::get_if<TokenInput>(&inputs.language)) {
                const FixedPrompt & fixed = *fixed_prompt_;
                const bool matches = tokens->token_ids.size == fixed.token_ids.size() &&
                    tokens->attention_mask.size == fixed.attention_mask.size() &&
                    std::equal(fixed.token_ids.begin(), fixed.token_ids.end(),
                               tokens->token_ids.data) &&
                    std::equal(fixed.attention_mask.begin(), fixed.attention_mask.end(),
                               tokens->attention_mask.data);
                if (!matches) {
                    throw Error(ErrorCode::failed_precondition,
                                "token input does not match the configured fixed prompt");
                }
            }
        }
        PreparedInputs prepared = prepare_inputs(inputs, *artifact_,
                                                 options_.enable_prefix_cache, random_);
        std::lock_guard<std::mutex> lock(runtime_->mutex);
        activate_session(*runtime_, id_);
        return run_reference_prediction(*runtime_, prepared, *artifact_);
    }

    Status reset() override {
        std::lock_guard<std::mutex> lock(runtime_->mutex);
        reset_session_cache(*runtime_, id_);
        random_.seed(0);
        ++generation_;
        return Status::success();
    }

private:
    std::shared_ptr<Runtime> runtime_;
    std::shared_ptr<const ArtifactContract> artifact_;
    SessionOptions options_;
    std::uint64_t id_ = 0;
    std::uint64_t generation_ = 0;
    LanguageEncoderPolicy language_policy_ = LanguageEncoderPolicy::resident;
    std::optional<FixedPrompt> fixed_prompt_;
    std::mt19937 random_;
};

class Gwp05Model final : public ModelImpl {
public:
    Gwp05Model(ModelInfo info, ArtifactContract artifact, const ModelOptions & options)
        : ModelImpl(std::move(info)), artifact_(
              std::make_shared<ArtifactContract>(std::move(artifact))),
          runtime_(std::make_shared<Runtime>()) {
        info_.artifact_policy = artifact_->policy;
        info_.artifact_components = artifact_->components;
        info_.language_encoder_policy = options.language_encoder_policy;
        info_.capabilities.backends.clear();
        info_.capabilities.backends.reserve(2);
        info_.capabilities.backends.push_back(Backend::cpu_metadata);
#if WAM_HAS_CUDA
        info_.capabilities.backends.push_back(Backend::cuda);
#endif
        info_.capabilities.precisions.clear();
        info_.capabilities.precisions.push_back(info_.precision);

        switch (options.language_encoder_policy) {
            case LanguageEncoderPolicy::resident:
                if (options.fixed_prompt) {
                    invalid("resident policy cannot configure a fixed prompt",
                            "fixed_prompt", "unexpected");
                }
                break;
            case LanguageEncoderPolicy::fixed:
                if (!options.fixed_prompt || options.fixed_prompt->token_ids.empty() ||
                    options.fixed_prompt->attention_mask.size() !=
                        options.fixed_prompt->token_ids.size()) {
                    invalid("fixed policy requires equal nonempty token and mask arrays",
                            "fixed_prompt", "missing or shape mismatch");
                }
                (void) semantics::prepare_prompt(options.fixed_prompt->token_ids,
                                                 options.fixed_prompt->attention_mask,
                                                 artifact_->geometry.t5_vocab_size);
                fixed_prompt_ = options.fixed_prompt;
                break;
            case LanguageEncoderPolicy::external_embedding:
                if (options.fixed_prompt) {
                    invalid("external_embedding policy cannot configure a fixed prompt",
                            "fixed_prompt", "unexpected");
                }
                break;
            default:
                invalid("unknown language encoder policy", "language_encoder_policy",
                        std::to_string(static_cast<std::uint32_t>(
                            options.language_encoder_policy)));
        }

        if (info_.backend == Backend::cuda) {
            if (options.device_index < 0) {
                invalid("CUDA device index cannot be negative", "device_index",
                        std::to_string(options.device_index));
            }
            const engine::KernelDispatch dispatch = engine::resolve_kernel_dispatch(
                info_.precision, info_.backend, artifact_->policy);
#ifndef WAM_GWP05_CUDNN
            if (dispatch.bf16_vae) {
                throw Error(ErrorCode::unsupported,
                            "the BF16 latency artifact requires WAM_CUDNN=ON");
            }
#endif
            engine::EngineOptions engine_options;
            engine_options.device_index = options.device_index;
            engine_options.prompt_cache_capacity = options.prompt_cache_capacity;
            engine_options.language_encoder_policy = options.language_encoder_policy;
            engine_options.dispatch = dispatch;
            if (options.fixed_prompt) {
                engine_options.fixed_tokens = options.fixed_prompt->token_ids;
                engine_options.fixed_attention_mask = options.fixed_prompt->attention_mask;
            }
            runtime_->engine = engine::create(*artifact_, engine_options);
            if (!runtime_->engine) {
                throw Error(ErrorCode::resource_exhausted,
                            "cannot initialize GWP F32 reference engine");
            }
            info_.capabilities.action = true;
            info_.capabilities.raw_images = true;
            info_.capabilities.token_input = true;
            info_.capabilities.precomputed_embedding = true;
            info_.capabilities.explicit_noise = true;
            info_.capabilities.token_input =
                options.language_encoder_policy != LanguageEncoderPolicy::external_embedding;
            info_.capabilities.arbitrary_token_input =
                options.language_encoder_policy == LanguageEncoderPolicy::resident;
            info_.capabilities.fixed_token_input =
                options.language_encoder_policy == LanguageEncoderPolicy::fixed;
            info_.text_encoder_resident = runtime_->engine->text_encoder_resident;
            info_.resident_device_bytes = runtime_->engine->resident_device_bytes;
            info_.peak_component_device_bytes =
                runtime_->engine->peak_component_device_bytes;
            info_.runtime_components = runtime_->engine->runtime_components;
        }
    }

    std::unique_ptr<SessionImpl> create_session(const SessionOptions & options) override {
        std::lock_guard<std::mutex> lock(runtime_->mutex);
        const std::uint64_t id = runtime_->next_session++;
        return std::make_unique<Gwp05Session>(runtime_, artifact_, options, id,
                                              info_.language_encoder_policy,
                                              fixed_prompt_);
    }

private:
    std::shared_ptr<ArtifactContract> artifact_;
    std::shared_ptr<Runtime> runtime_;
    std::optional<FixedPrompt> fixed_prompt_;
};

} // namespace

PreparedInputs prepare_inputs(const Inputs & inputs, const ArtifactContract & artifact,
                              bool enable_prefix_cache, std::mt19937 & random) {
    PreparedInputs prepared;
    prepared.view.enable_prefix_cache = enable_prefix_cache;
    const auto real_state = static_cast<std::int64_t>(artifact.geometry.real_state_dim);
    const auto padded_state = static_cast<std::int64_t>(artifact.geometry.action_dim);
    prepared.state = copy_f32_tensor(inputs.state, {{real_state}, {padded_state}}, "state");
    prepared.view.state = prepared.state.data();

    const auto chunk = static_cast<std::int64_t>(artifact.geometry.action_chunk);
    const auto action_dim = static_cast<std::int64_t>(artifact.geometry.action_dim);
    if (inputs.noise.empty()) {
        prepared.noise.resize(static_cast<std::size_t>(chunk * action_dim));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float & value : prepared.noise) value = normal(random);
    } else {
        prepared.noise = copy_f32_tensor(
            inputs.noise, {{chunk, action_dim}, {1, chunk, action_dim}}, "noise");
    }
    prepared.view.noise = prepared.noise.data();
    prepare_vision(inputs, artifact, prepared);
    prepare_language(inputs, artifact, prepared);
    return prepared;
}

void activate_session(Runtime & runtime, std::uint64_t session_id) {
    if (runtime.active_session == session_id) return;
    if (runtime.engine) runtime.engine->reset();
    runtime.active_session = session_id;
}

void reset_session_cache(Runtime & runtime, std::uint64_t session_id) {
    if (runtime.active_session != session_id) return;
    if (runtime.engine) runtime.engine->reset();
    runtime.active_session = 0;
}

Prediction run_reference_prediction(Runtime & runtime, PreparedInputs & inputs,
                                    const ArtifactContract & artifact) {
    if (!runtime.engine) {
        throw Error(ErrorCode::unsupported,
                    "GWP prediction requires an initialized F32 compute backend");
    }
    std::vector<float> values = runtime.engine->predict(inputs.view);
    const std::size_t expected = static_cast<std::size_t>(artifact.geometry.action_chunk) *
        artifact.geometry.real_action_dim;
    if (values.size() != expected) {
        throw Error(ErrorCode::inference_failed, "GWP reference graph execution failed",
                    {{"action.elements", std::to_string(values.size())}});
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw Error(ErrorCode::inference_failed, "GWP prediction contains NaN or Inf",
                        {{"action", std::to_string(index)}});
        }
    }

    Prediction prediction;
    prediction.action.dtype = DType::f32;
    prediction.action.shape = {
        static_cast<std::int64_t>(artifact.geometry.action_chunk),
        static_cast<std::int64_t>(artifact.geometry.real_action_dim),
    };
    prediction.action.layout = "T,A";
    prediction.action.byte_order = ByteOrder::little;
    prediction.action.data.resize(values.size() * sizeof(float));
    std::memcpy(prediction.action.data.data(), values.data(), prediction.action.data.size());

    const engine::EngineTelemetry & source = runtime.engine->stats;
    Stats & stats = prediction.stats;
    stats.preprocess_milliseconds = source.ms_vae_preprocess;
    stats.vae_milliseconds = source.ms_vae;
    stats.text_encoder_milliseconds = source.ms_umt5;
    stats.prompt_projection_milliseconds = source.ms_prompt_projection;
    stats.prefix_milliseconds = source.ms_prefix_cache;
    stats.denoise_milliseconds = source.ms_denoise;
    stats.total_milliseconds = source.ms_total;
    stats.postprocess_milliseconds = std::max(
        0.0, static_cast<double>(source.ms_total - source.ms_vision - source.ms_inference));
    stats.denoise_step_milliseconds.assign(source.ms_denoise_steps.begin(),
                                            source.ms_denoise_steps.end());
    stats.additional_timings = {
        {"vae_graph", source.ms_vae_graph},
        {"prefix_graph_build", source.ms_prefix_graph_build},
        {"action_graph_build", source.ms_action_graph_build},
    };
    stats.prompt_cache_hit = source.prompt_cache_hit;
    stats.projected_prompt_cache_hit = source.projected_prompt_cache_hit;
    return prediction;
}

std::unique_ptr<ModelImpl> create_model(ModelInfo info, ArtifactContract artifact,
                                        const ModelOptions & options) {
    return std::make_unique<Gwp05Model>(std::move(info), std::move(artifact), options);
}

} // namespace wam::internal::gwp05

namespace wam::internal {

void register_gwp05(ModelRegistry & registry) {
    registry.add(Arch::gwp05,
        [](const ModelOptions & options, ModelInfo info,
           std::shared_ptr<GgufReader> reader) {
            auto artifact = gwp05::inspect_artifact(std::move(reader), options);
            return gwp05::create_model(std::move(info), std::move(artifact), options);
        });
}

} // namespace wam::internal
