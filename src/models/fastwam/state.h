#pragma once

#include "backends/ggml/backend_context.h"
#include "backends/ggml/debug_dump.h"
#include "backends/ggml/weight_store.h"
#include "runtime/logger.h"
#include "wam/model.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace wam::internal::fastwam {

struct FastWamContract;

class ModelResources final {
public:
    ModelResources(std::shared_ptr<runtime::Logger> logger,
                   std::shared_ptr<ggml_backend::DebugDump> debug_dump);

    bool initialize(const FastWamContract & contract, int device_index);
    ggml_backend_t backend() const noexcept;
    ggml_tensor * weight(const char * name) const noexcept;
    const ggml_backend::DebugDump & debug_dump() const noexcept;

    std::uint64_t resident_device_bytes = 0;
    std::vector<RuntimeComponentInfo> runtime_components;
    std::mutex execution_mutex;

private:
    std::shared_ptr<runtime::Logger> logger_;
    std::shared_ptr<ggml_backend::DebugDump> debug_dump_;
    ggml_backend::BackendContext backend_context_;
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_backend::WeightStore> weights_;
};

struct SessionState final {
    void reset() noexcept { prediction_count = 0; }

    std::uint64_t prediction_count = 0;
};

} // namespace wam::internal::fastwam
