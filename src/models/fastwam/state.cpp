#include "models/fastwam/state.h"

#include "artifact/gguf_reader.h"
#include "models/fastwam/contract.h"

#include "ggml.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace wam::internal::fastwam {

ModelResources::ModelResources(
    std::shared_ptr<runtime::Logger> logger,
    std::shared_ptr<ggml_backend::DebugDump> debug_dump)
    : logger_(std::move(logger)), debug_dump_(std::move(debug_dump)) {}

ggml_backend_t ModelResources::backend() const noexcept {
    return backend_;
}

ggml_tensor * ModelResources::weight(const char * name) const noexcept {
    return weights_ == nullptr ? nullptr : weights_->find(name);
}

const ggml_backend::DebugDump & ModelResources::debug_dump() const noexcept {
    return *debug_dump_;
}

bool ModelResources::initialize(const FastWamContract & contract,
                                int device_index) {
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
    const std::vector<GgufTensorInfo> & tensors = contract.reader->tensors();
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
        if (!contract.reader->read_tensor(info.name, staging.data(), bytes)) {
            return false;
        }
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

} // namespace wam::internal::fastwam
