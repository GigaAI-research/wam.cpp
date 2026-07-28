#include "backends/ggml/backend_context.h"

#include <utility>

namespace wam::internal::ggml_backend {

BackendContext::BackendContext(ggml_backend_t backend) noexcept
    : backend_(backend) {}

BackendContext::~BackendContext() { reset(); }

BackendContext::BackendContext(BackendContext && other) noexcept
    : backend_(other.release()) {}

BackendContext & BackendContext::operator=(BackendContext && other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
}

ggml_backend_t BackendContext::get() const noexcept { return backend_; }

BackendContext::operator bool() const noexcept { return backend_ != nullptr; }

void BackendContext::reset(ggml_backend_t backend) noexcept {
    if (backend_ == backend) return;
    if (backend_ != nullptr) ggml_backend_free(backend_);
    backend_ = backend;
}

ggml_backend_t BackendContext::release() noexcept {
    return std::exchange(backend_, nullptr);
}

} // namespace wam::internal::ggml_backend
