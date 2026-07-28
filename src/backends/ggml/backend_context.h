#pragma once

#include "ggml-backend.h"

namespace wam::internal::ggml_backend {

class BackendContext final {
public:
    BackendContext() = default;
    explicit BackendContext(ggml_backend_t backend) noexcept;
    ~BackendContext();
    BackendContext(BackendContext && other) noexcept;
    BackendContext & operator=(BackendContext && other) noexcept;
    BackendContext(const BackendContext &) = delete;
    BackendContext & operator=(const BackendContext &) = delete;

    ggml_backend_t get() const noexcept;
    explicit operator bool() const noexcept;
    void reset(ggml_backend_t backend = nullptr) noexcept;
    ggml_backend_t release() noexcept;

private:
    ggml_backend_t backend_ = nullptr;
};

} // namespace wam::internal::ggml_backend
