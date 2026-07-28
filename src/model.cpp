#include "wam/model.h"

#include "arch.h"
#include "model_internal.h"
#include "model_registry.h"
#include "models/common/gguf_reader.h"
#include "policy/policy_spec.h"
#include "wam/error.h"
#include "wam/session.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace wam {
namespace {

[[noreturn]] void throw_moved_from(const char * type) {
    throw Error(ErrorCode::failed_precondition,
                std::string(type) + " object has been moved from");
}

template <typename Function>
decltype(auto) translate_internal_errors(const char * operation,
                                         Function && function) {
    try {
        return std::forward<Function>(function)();
    } catch (const Error &) {
        throw;
    } catch (const std::exception & error) {
        throw Error(ErrorCode::internal,
                    std::string(operation) + " failed: " + error.what());
    } catch (...) {
        throw Error(ErrorCode::internal,
                    std::string(operation) + " failed with an unknown error");
    }
}

bool valid_backend(Backend backend) noexcept {
    switch (backend) {
        case Backend::automatic:
        case Backend::cpu:
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

void validate_runtime_config(const RuntimeConfig & config) {
    if (!valid_backend(config.backend)) {
        throw Error(ErrorCode::invalid_argument,
                    "backend is unknown or invalid",
                    {{"backend", "unsupported enum value"}});
    }
    if (!valid_precision(config.compute_precision)) {
        throw Error(ErrorCode::invalid_argument,
                    "compute_precision is unknown or invalid",
                    {{"compute_precision", "unsupported enum value"}});
    }
    if (config.device_index < 0) {
        throw Error(ErrorCode::invalid_argument,
                    "device_index must be non-negative",
                    {{"device_index", "negative"}});
    }
    if (!valid_language_mode(config.language_mode)) {
        throw Error(ErrorCode::invalid_argument,
                    "language_mode is invalid",
                    {{"language_mode", "unsupported enum value"}});
    }
    if (config.debug_dump.enabled && config.debug_dump.directory.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "debug dump directory must not be empty when enabled",
                    {{"debug_dump.directory", "empty"}});
    }
    if (config.fixed_prompt.has_value()) {
        const FixedPrompt & prompt = *config.fixed_prompt;
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

Model::Model(std::shared_ptr<internal::ModelImpl> impl)
    : impl_(std::move(impl)) {}

Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

Model Model::load(const std::string & artifact_path,
                  const RuntimeConfig & config) {
    if (artifact_path.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "artifact_path must not be empty",
                    {{"artifact_path", "empty"}});
    }
    validate_runtime_config(config);

    std::shared_ptr<internal::GgufReader> reader =
        internal::GgufReader::open(artifact_path);
    const internal::Arch architecture = detect_architecture(*reader);
    std::optional<PolicySpec> policy_spec =
        internal::policy::try_read_policy_spec(*reader);

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
    info.artifact_path = artifact_path;
    info.artifact_bytes = reader->file_size();
    info.backend = config.backend;
    info.compute_precision = config.compute_precision;
    info.language_mode = config.language_mode;

    std::unique_ptr<internal::ModelImpl> impl = translate_internal_errors(
        "model creation", [&] {
            return (*factory)(config, std::move(info), std::move(policy_spec),
                              std::move(reader));
        });
    return internal::adopt_model(std::move(impl));
}

Session Model::create_session(const SessionConfig & config) const {
    if (!impl_) throw_moved_from("Model");
    std::unique_ptr<internal::SessionImpl> session = translate_internal_errors(
        "session creation", [&] { return impl_->create_session(config); });
    if (!session) {
        throw Error(ErrorCode::internal,
                    "model returned a null session implementation");
    }
    return Session(impl_, std::move(session));
}

const ModelInfo & Model::info() const {
    if (!impl_) throw_moved_from("Model");
    return impl_->info();
}

Session::Session(std::shared_ptr<internal::ModelImpl> model,
                 std::unique_ptr<internal::SessionImpl> impl)
    : model_(std::move(model)), impl_(std::move(impl)) {}

Session::~Session() = default;
Session::Session(Session &&) noexcept = default;
Session & Session::operator=(Session &&) noexcept = default;

Prediction Session::predict(const Observation & observation) {
    if (!impl_) throw_moved_from("Session");
    return translate_internal_errors(
        "prediction", [&] { return impl_->predict(observation); });
}

void Session::reset() {
    if (!impl_) throw_moved_from("Session");
    translate_internal_errors("session reset", [&] { impl_->reset(); });
}

namespace internal {

Model ModelAccess::adopt(std::unique_ptr<ModelImpl> impl) {
    if (!impl) {
        throw Error(ErrorCode::internal,
                    "cannot adopt a null model implementation");
    }
    return Model(std::shared_ptr<ModelImpl>(std::move(impl)));
}

const ModelImpl & ModelAccess::impl(const Model & model) {
    if (!model.impl_) throw_moved_from("Model");
    return *model.impl_;
}

Model adopt_model(std::unique_ptr<ModelImpl> impl) {
    return ModelAccess::adopt(std::move(impl));
}

const ModelImpl & model_impl(const Model & model) {
    return ModelAccess::impl(model);
}

} // namespace internal

} // namespace wam
