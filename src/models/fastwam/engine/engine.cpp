#include "engine.h"
#include "engine_internal.h"
#include "pipeline.h"

#include "models/common/gguf_reader.h"

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace wam::internal::fastwam {
namespace {

class Bf16Engine final : public Engine {
public:
    ggml_backend_t backend() const noexcept override { return backend_; }

    ggml_tensor * weight(const char * name) const noexcept override {
        const auto found = weights_.find(name);
        return found == weights_.end() ? nullptr : found->second;
    }

    ~Bf16Engine() override {
        if (weight_buffer_) ggml_backend_buffer_free(weight_buffer_);
        if (weight_context_) ggml_free(weight_context_);
        if (backend_) ggml_backend_free(backend_);
    }

    bool initialize(const ArtifactContract & artifact, int device_index) {
#ifdef GGML_USE_CUDA
        backend_ = ggml_backend_cuda_init(device_index);
#else
        (void) device_index;
#endif
        if (!backend_) {
            std::fprintf(stderr,
                         "wam(fastwam): native BF16 requires a CUDA build; fallback is disabled\n");
            return false;
        }
        const char * backend_name = ggml_backend_name(backend_);
        std::fprintf(stderr,
                     "wam(fastwam): execution=native-bf16 backend=%s "
                     "weights=BF16 sensitive_math=F32 fallback=none\n",
                     backend_name ? backend_name : "unknown");

        const auto begin = std::chrono::steady_clock::now();
        const std::vector<GgufTensorInfo> & tensors = artifact.reader->tensors();
        ggml_init_params params{};
        params.mem_size = (tensors.size() + 64) * ggml_tensor_overhead();
        params.no_alloc = true;
        weight_context_ = ggml_init(params);
        if (!weight_context_) return false;

        std::size_t largest = 0;
        for (const GgufTensorInfo & info : tensors) {
            const ggml_type type = info.dtype == DType::bf16
                ? GGML_TYPE_BF16
                : info.dtype == DType::f32 ? GGML_TYPE_F32 : GGML_TYPE_COUNT;
            if (type == GGML_TYPE_COUNT || info.shape.empty()) return false;
            ggml_tensor * destination = ggml_new_tensor(
                weight_context_, type, static_cast<int>(info.shape.size()),
                info.shape.data());
            ggml_set_name(destination, info.name.c_str());
            weights_.emplace(info.name, destination);
            largest = std::max(largest, ggml_nbytes(destination));
        }
        weight_buffer_ = ggml_backend_alloc_ctx_tensors(weight_context_, backend_);
        if (!weight_buffer_) return false;

        std::vector<std::uint8_t> staging(largest);
        std::size_t loaded = 0;
        for (const GgufTensorInfo & info : tensors) {
            ggml_tensor * destination = weights_.at(info.name);
            const std::size_t bytes = ggml_nbytes(destination);
            if (!artifact.reader->read_tensor(info.name, staging.data(), bytes)) return false;
            ggml_backend_tensor_set(destination, staging.data(), 0, bytes);
            if ((++loaded % 100) == 0) {
                std::fprintf(stderr, "wam(fastwam): loaded %zu/%zu tensors\r",
                             loaded, tensors.size());
                std::fflush(stderr);
            }
        }
        ggml_backend_synchronize(backend_);
        resident_device_bytes = ggml_backend_buffer_get_size(weight_buffer_);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        runtime_components.push_back({"fastwam-bf16", true,
                                      resident_device_bytes, milliseconds, 0.0});
        std::fprintf(stderr,
                     "wam(fastwam): loaded %zu tensors (%.2f GiB) in %.1f ms\n",
                     tensors.size(), static_cast<double>(resident_device_bytes) /
                         (1024.0 * 1024.0 * 1024.0), milliseconds);
        return true;
    }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> weights_;
};

} // namespace

std::unique_ptr<Engine> create_bf16_engine(const ArtifactContract & artifact,
                                           int device_index) {
    auto engine = std::make_unique<Bf16Engine>();
    if (!engine->initialize(artifact, device_index)) return nullptr;
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
        create_bf16_engine(artifact, options.device_index);
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
