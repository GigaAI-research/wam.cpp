#pragma once

#include "wam/observation.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wam::internal::ggml_backend {

class WeightStore final {
public:
    WeightStore(ggml_backend_t backend, std::size_t tensor_capacity);
    ~WeightStore();
    WeightStore(WeightStore &&) = delete;
    WeightStore & operator=(WeightStore &&) = delete;
    WeightStore(const WeightStore &) = delete;
    WeightStore & operator=(const WeightStore &) = delete;

    ggml_tensor * define(const std::string & name, DType dtype,
                         const std::vector<std::int64_t> & shape);
    void allocate();
    void set(const std::string & name, const void * data, std::size_t bytes);
    ggml_tensor * find(const std::string & name) const noexcept;
    std::uint64_t bytes() const noexcept;
    std::size_t size() const noexcept;

private:
    ggml_backend_t backend_ = nullptr;
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};

} // namespace wam::internal::ggml_backend
