#include "wam/wam.h"

#include "arch.h"
#include "model_internal.h"
#include "model_registry.h"
#include "models/common/gguf_reader.h"

#include <memory>
#include <string>
#include <utility>

namespace wam {

struct Model {
    std::unique_ptr<internal::ModelImpl> impl;
    std::size_t active_sessions = 0;
    bool handle_released = false;
};

struct Session {
    Model * owner = nullptr;
    std::unique_ptr<internal::SessionImpl> impl;
};

namespace {

internal::Arch detect_architecture(const internal::GgufReader & reader) {
    const std::string value = reader.require_string("general.architecture");
    const internal::Arch arch = internal::arch_from_name(value);
    if (arch == internal::Arch::unknown) {
        throw Error(ErrorCode::unsupported, "unsupported GGUF architecture",
                    {{"general.architecture", value}});
    }
    return arch;
}

Backend resolve_backend(Backend requested) {
    if (requested == Backend::unknown) {
        throw Error(ErrorCode::invalid_argument, "model backend cannot be unknown");
    }
    if (requested == Backend::automatic) {
        return Backend::cpu_metadata;
    }
    if (requested == Backend::cuda) {
#if WAM_HAS_CUDA
        return Backend::cuda;
#else
        throw Error(ErrorCode::unsupported, "CUDA backend was not compiled");
#endif
    }
    return requested;
}

void destroy_if_released(Model * model) noexcept {
    if (model != nullptr && model->handle_released && model->active_sessions == 0) {
        delete model;
    }
}

} // namespace

Model * model_load(const ModelOptions & options) {
    if (options.artifact_path.empty()) {
        throw Error(ErrorCode::invalid_argument, "artifact_path is required");
    }
    if (options.precision == Precision::unknown) {
        throw Error(ErrorCode::invalid_argument, "precision cannot be unknown");
    }

    std::shared_ptr<internal::GgufReader> reader =
        internal::GgufReader::open(options.artifact_path);
    const internal::Arch arch = detect_architecture(*reader);
    const internal::RegistryEntry * entry = internal::model_registry().find(arch);
    if (entry == nullptr) {
        throw Error(ErrorCode::unsupported,
                    "architecture is recognized but not compiled",
                    {{"general.architecture", std::string(internal::arch_name(arch))}});
    }

    ModelInfo info;
    info.architecture = std::string(internal::arch_name(arch));
    info.artifact_path = options.artifact_path;
    info.artifact_bytes = reader->file_size();
    info.backend = resolve_backend(options.backend);
    info.precision = options.precision;

    auto model = std::make_unique<Model>();
    model->impl = entry->factory(options, std::move(info), std::move(reader));
    if (!model->impl) {
        throw Error(ErrorCode::internal, "model factory returned null");
    }
    return model.release();
}

Session * session_create(Model * model, const SessionOptions & options) {
    if (model == nullptr || model->handle_released || !model->impl) {
        throw Error(ErrorCode::invalid_argument, "valid model is required");
    }
    auto session = std::make_unique<Session>();
    session->owner = model;
    session->impl = model->impl->create_session(options);
    if (!session->impl) {
        throw Error(ErrorCode::internal, "session factory returned null");
    }
    ++model->active_sessions;
    return session.release();
}

Prediction predict(Session * session, const Inputs & inputs) {
    if (session == nullptr || !session->impl) {
        throw Error(ErrorCode::invalid_argument, "valid session is required");
    }
    return session->impl->predict(inputs);
}

Status session_reset(Session * session) {
    if (session == nullptr || !session->impl) {
        return {ErrorCode::invalid_argument, "valid session is required", {}};
    }
    try {
        return session->impl->reset();
    } catch (const Error & error) {
        return {error.code(), error.what(), error.details()};
    } catch (const std::exception & error) {
        return {ErrorCode::internal, error.what(), {}};
    } catch (...) {
        return {ErrorCode::internal, "unknown session reset failure", {}};
    }
}

const ModelInfo & model_info(const Model * model) {
    if (model == nullptr || model->handle_released || !model->impl) {
        throw Error(ErrorCode::invalid_argument, "valid model is required");
    }
    return model->impl->info();
}

void session_free(Session * session) noexcept {
    if (session == nullptr) return;
    Model * owner = session->owner;
    delete session;
    if (owner != nullptr && owner->active_sessions > 0) {
        --owner->active_sessions;
    }
    destroy_if_released(owner);
}

void model_free(Model * model) noexcept {
    if (model == nullptr || model->handle_released) return;
    model->handle_released = true;
    destroy_if_released(model);
}

} // namespace wam
