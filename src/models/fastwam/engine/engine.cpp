#include "engine.h"
#include "engine_internal.h"
#include "pipeline.h"

#include "artifact/gguf_reader.h"
#include "backends/ggml/backend_context.h"
#include "backends/ggml/weight_store.h"
#include "wam/error.h"

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace wam::internal::fastwam {
namespace {

class Bf16Engine final : public Engine {
public:
    Bf16Engine(std::shared_ptr<runtime::Logger> logger,
               std::shared_ptr<ggml_backend::DebugDump> debug_dump)
        : logger_(std::move(logger)), debug_dump_(std::move(debug_dump)) {}

    ggml_backend_t backend() const noexcept override { return backend_; }

    ggml_tensor * weight(const char * name) const noexcept override {
        return weights_ == nullptr ? nullptr : weights_->find(name);
    }

    const ggml_backend::DebugDump & debug_dump() const noexcept override {
        return *debug_dump_;
    }

    bool initialize(const ArtifactContract & artifact, int device_index) {
#ifdef GGML_USE_CUDA
        backend_context_.reset(ggml_backend_cuda_init(device_index));
#else
        (void) device_index;
#endif
        backend_ = backend_context_.get();
        if (!backend_) {
            logger_->log(LogLevel::error,
                         "fastwam: native BF16 requires a CUDA build; fallback is disabled");
            return false;
        }
        const char * backend_name = ggml_backend_name(backend_);
        logger_->logf(LogLevel::info,
                      "fastwam: execution=native-bf16 backend=%s "
                      "weights=BF16 sensitive_math=F32 fallback=none",
                      backend_name ? backend_name : "unknown");

        const auto begin = std::chrono::steady_clock::now();
        const std::vector<GgufTensorInfo> & tensors = artifact.reader->tensors();
        weights_ = std::make_unique<ggml_backend::WeightStore>(
            backend_, tensors.size());

        std::size_t largest = 0;
        for (const GgufTensorInfo & info : tensors) {
            ggml_tensor * destination =
                weights_->define(info.name, info.dtype, info.shape);
            largest = std::max(largest, ggml_nbytes(destination));
        }
        weights_->allocate();

        std::vector<std::uint8_t> staging(largest);
        std::size_t loaded = 0;
        for (const GgufTensorInfo & info : tensors) {
            ggml_tensor * destination = weights_->find(info.name);
            const std::size_t bytes = ggml_nbytes(destination);
            if (!artifact.reader->read_tensor(info.name, staging.data(), bytes)) return false;
            weights_->set(info.name, staging.data(), bytes);
            if ((++loaded % 100) == 0) {
                logger_->logf(LogLevel::debug,
                              "fastwam: loaded %zu/%zu tensors",
                              loaded, tensors.size());
            }
        }
        ggml_backend_synchronize(backend_);
        resident_device_bytes = weights_->bytes();
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        runtime_components.push_back({"fastwam-bf16", true,
                                      resident_device_bytes, milliseconds, 0.0});
        logger_->logf(LogLevel::info,
                      "fastwam: loaded %zu tensors (%.2f GiB) in %.1f ms",
                      tensors.size(), static_cast<double>(resident_device_bytes) /
                          (1024.0 * 1024.0 * 1024.0), milliseconds);
        return true;
    }

private:
    std::shared_ptr<runtime::Logger> logger_;
    std::shared_ptr<ggml_backend::DebugDump> debug_dump_;
    ggml_backend::BackendContext backend_context_;
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_backend::WeightStore> weights_;
};

} // namespace

std::unique_ptr<Engine> create_bf16_engine(const ArtifactContract & artifact,
                                           const engine::EngineOptions & options) {
    auto logger = options.logger
        ? options.logger
        : std::make_shared<runtime::Logger>(RuntimeConfig{});
    auto debug_dump = options.debug_dump
        ? options.debug_dump
        : std::make_shared<ggml_backend::DebugDump>();
    auto engine = std::make_unique<Bf16Engine>(std::move(logger),
                                               std::move(debug_dump));
    if (!engine->initialize(artifact, options.device_index)) return nullptr;
    return engine;
}

} // namespace wam::internal::fastwam

namespace wam::internal::fastwam::engine {

class Engine final {
public:
    std::unique_ptr<fastwam::Engine> runtime;
    const ArtifactContract * artifact = nullptr;
    EngineInfo info;
    std::mutex execution_mutex;
};

class EngineSession final {
public:
    explicit EngineSession(Engine & owner) : owner_(&owner) {}

private:
    friend CoreAction predict(EngineSession &, const PreparedInputs &);
    friend void reset(EngineSession &);

    Engine * owner_ = nullptr;
};

void EngineDeleter::operator()(Engine * value) const noexcept { delete value; }
void EngineSessionDeleter::operator()(EngineSession * value) const noexcept {
    delete value;
}

EnginePtr create_engine(const ArtifactContract & artifact,
                        const EngineOptions & options) {
    if (options.backend != Backend::automatic &&
        options.backend != Backend::cuda) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM execution requires the CUDA backend");
    }
    if (options.compute_precision != ComputePrecision::automatic &&
        options.compute_precision != ComputePrecision::bf16) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM Gate B supports BF16 compute only");
    }
    std::unique_ptr<fastwam::Engine> runtime =
        create_bf16_engine(artifact, options);
    if (!runtime) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot initialize FastWAM BF16 CUDA engine");
    }
    EnginePtr result(new Engine());
    result->info.backend = Backend::cuda;
    result->info.compute_precision = ComputePrecision::bf16;
    result->info.resident_device_bytes = runtime->resident_device_bytes;
    result->info.peak_component_device_bytes = runtime->resident_device_bytes;
    result->info.runtime_components = runtime->runtime_components;
    result->artifact = &artifact;
    result->runtime = std::move(runtime);
    return result;
}

EngineSessionPtr create_engine_session(
    Engine & instance, const EngineSessionOptions &) {
    if (!instance.runtime || instance.artifact == nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "FastWAM engine has no model resources");
    }
    return EngineSessionPtr(new EngineSession(instance));
}

CoreAction predict(EngineSession & session, const PreparedInputs & inputs) {
    if (session.owner_ == nullptr || !session.owner_->runtime ||
        session.owner_->artifact == nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "FastWAM engine session is not initialized");
    }
    Engine & engine = *session.owner_;
    std::lock_guard<std::mutex> lock(engine.execution_mutex);
    return run_pipeline(*engine.runtime, *engine.artifact, inputs);
}

void reset(EngineSession & session) {
    if (session.owner_ == nullptr) {
        throw Error(ErrorCode::failed_precondition,
                    "FastWAM engine session is not initialized");
    }
    std::lock_guard<std::mutex> lock(session.owner_->execution_mutex);
}

const EngineInfo & engine_info(const Engine & engine) noexcept {
    return engine.info;
}

} // namespace wam::internal::fastwam::engine
