#include "wam/wam.h"

#include "arch.h"
#include "model_internal.h"
#include "model_registry.h"
#include "models/common/gguf_reader.h"
#include "policy/policy_spec.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace wam {

struct Model {
    explicit Model(std::unique_ptr<internal::ModelImpl> model_impl)
        : impl(std::move(model_impl)) {}

    std::unique_ptr<internal::ModelImpl> impl;
    std::size_t active_sessions = 0;
    bool handle_released = false;
};

struct Session {
    Model * owner = nullptr;
    std::unique_ptr<internal::SessionImpl> impl;
};

namespace {

[[noreturn]] void throw_invalid_handle(const char * handle_name) {
    throw Error(ErrorCode::invalid_argument,
                std::string(handle_name) + " handle is null or released");
}

bool valid_backend(Backend backend) noexcept {
    switch (backend) {
        case Backend::automatic:
        case Backend::cuda:
        case Backend::cpu_metadata:
            return true;
        case Backend::unknown:
            return false;
    }
    return false;
}

bool valid_precision(ComputePrecision precision) noexcept {
    switch (precision) {
        case ComputePrecision::automatic:
        case ComputePrecision::f32:
        case ComputePrecision::f16:
        case ComputePrecision::bf16:
        case ComputePrecision::fp8_e4m3:
        case ComputePrecision::fp8_e5m2:
        case ComputePrecision::int8:
            return true;
        case ComputePrecision::unknown:
            return false;
    }
    return false;
}

bool valid_language_mode(LanguageRuntimeMode mode) noexcept {
    switch (mode) {
        case LanguageRuntimeMode::automatic:
        case LanguageRuntimeMode::tokens:
        case LanguageRuntimeMode::external_embedding:
            return true;
    }
    return false;
}

void validate_model_options(const ModelOptions & options) {
    if (options.artifact_path.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "artifact_path must not be empty",
                    {{"artifact_path", "empty"}});
    }
    if (!valid_backend(options.backend)) {
        throw Error(ErrorCode::invalid_argument,
                    "backend is unknown or invalid",
                    {{"backend", "unsupported enum value"}});
    }
    if (!valid_precision(options.compute_precision)) {
        throw Error(ErrorCode::invalid_argument,
                    "compute_precision is unknown or invalid",
                    {{"compute_precision", "unsupported enum value"}});
    }
    if (options.device_index < 0) {
        throw Error(ErrorCode::invalid_argument,
                    "device_index must be non-negative",
                    {{"device_index", "negative"}});
    }
    if (!valid_language_mode(options.language_mode)) {
        throw Error(ErrorCode::invalid_argument,
                    "language_mode is invalid",
                    {{"language_mode", "unsupported enum value"}});
    }
    if (options.fixed_prompt.has_value()) {
        const FixedPrompt & prompt = *options.fixed_prompt;
        if (prompt.token_ids.empty()) {
            throw Error(ErrorCode::invalid_argument,
                        "fixed_prompt token_ids must not be empty",
                        {{"fixed_prompt.token_ids", "empty"}});
        }
        if (prompt.token_ids.size() != prompt.attention_mask.size()) {
            throw Error(
                ErrorCode::invalid_argument,
                "fixed_prompt token_ids and attention_mask sizes differ",
                {{"fixed_prompt.attention_mask", "size mismatch"}});
        }
    }
}

internal::Arch detect_architecture(const internal::GgufReader & reader) {
    const std::string value = reader.require_string("general.architecture");
    const internal::Arch architecture = internal::arch_from_name(value);
    if (architecture == internal::Arch::unknown) {
        throw Error(ErrorCode::unsupported, "unsupported GGUF architecture",
                    {{"general.architecture", value}});
    }
    return architecture;
}

} // namespace

Model * model_load(const ModelOptions & options) {
    validate_model_options(options);

    std::shared_ptr<internal::GgufReader> reader =
        internal::GgufReader::open(options.artifact_path);
    const internal::Arch architecture = detect_architecture(*reader);
    std::optional<internal::policy::PolicySpecDraft> policy_spec =
        internal::policy::try_read_policy_spec_draft(*reader);

    const internal::ModelFactory * factory =
        internal::model_registry().find(architecture);
    if (factory == nullptr) {
        throw Error(ErrorCode::unsupported,
                    "GGUF architecture is recognized but not compiled",
                    {{"general.architecture",
                      std::string(internal::arch_name(architecture))}});
    }

    ModelInfo info;
    info.architecture = std::string(internal::arch_name(architecture));
    info.artifact_path = options.artifact_path;
    info.artifact_bytes = reader->file_size();
    info.backend = options.backend;
    info.compute_precision = options.compute_precision;
    info.language_mode = options.language_mode;
    if (policy_spec.has_value()) {
        info.artifact_policy = policy_spec->identity.profile;
    }

    std::unique_ptr<internal::ModelImpl> impl =
        (*factory)(options, std::move(info), std::move(policy_spec),
                   std::move(reader));
    if (impl == nullptr) {
        throw Error(ErrorCode::internal,
                    "model factory returned a null implementation");
    }
    return internal::adopt_model(std::move(impl));
}

Session * session_create(Model * model, const SessionOptions & options) {
    if (model == nullptr || model->handle_released || model->impl == nullptr) {
        throw_invalid_handle("model");
    }

    std::unique_ptr<internal::SessionImpl> impl =
        model->impl->create_session(options);
    if (impl == nullptr) {
        throw Error(ErrorCode::internal,
                    "model returned a null session implementation");
    }

    auto session = std::make_unique<Session>();
    session->owner = model;
    session->impl = std::move(impl);
    ++model->active_sessions;
    return session.release();
}

Prediction predict(Session * session, const Inputs & inputs) {
    if (session == nullptr || session->impl == nullptr) {
        throw_invalid_handle("session");
    }
    return session->impl->predict(inputs);
}

Status session_reset(Session * session) {
    if (session == nullptr || session->impl == nullptr) {
        return {ErrorCode::invalid_argument,
                "session handle is null or released", {}};
    }

    try {
        return session->impl->reset();
    } catch (const Error & error) {
        return {error.code(), error.what(), error.details()};
    } catch (const std::exception & error) {
        return {ErrorCode::internal, error.what(), {}};
    } catch (...) {
        return {ErrorCode::internal,
                "unknown exception while resetting session", {}};
    }
}

const ModelInfo & model_info(const Model * model) {
    if (model == nullptr || model->handle_released || model->impl == nullptr) {
        throw_invalid_handle("model");
    }
    return model->impl->info();
}

void session_free(Session * session) noexcept {
    if (session == nullptr) {
        return;
    }

    Model * owner = session->owner;
    delete session;

    if (owner == nullptr) {
        return;
    }
    if (owner->active_sessions > 0) {
        --owner->active_sessions;
    }
    if (owner->handle_released && owner->active_sessions == 0) {
        delete owner;
    }
}

void model_free(Model * model) noexcept {
    if (model == nullptr || model->handle_released) {
        return;
    }

    model->handle_released = true;
    if (model->active_sessions == 0) {
        delete model;
    }
}

namespace internal {

Model * adopt_model(std::unique_ptr<ModelImpl> impl) {
    if (impl == nullptr) {
        throw Error(ErrorCode::internal,
                    "cannot adopt a null model implementation");
    }
    return new Model(std::move(impl));
}

const ModelImpl & model_impl(const Model * model) {
    if (model == nullptr || model->handle_released || model->impl == nullptr) {
        throw_invalid_handle("model");
    }
    return *model->impl;
}

} // namespace internal

} // namespace wam
