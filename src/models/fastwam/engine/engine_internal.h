#pragma once

#include "wam/model.h"

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace wam::internal::fastwam {

class Engine {
public:
    virtual ~Engine() = default;
    virtual ggml_backend_t backend() const noexcept = 0;
    virtual ggml_tensor * weight(const char * name) const noexcept = 0;

    std::uint64_t resident_device_bytes = 0;
    std::vector<RuntimeComponentInfo> runtime_components;
};

} // namespace wam::internal::fastwam
