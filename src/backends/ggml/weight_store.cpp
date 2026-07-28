#include "backends/ggml/weight_store.h"

#include "wam/error.h"

#include <algorithm>
#include <limits>

namespace wam::internal::ggml_backend {
namespace {

ggml_type to_ggml_type(DType dtype) {
    switch (dtype) {
        case DType::f32: return GGML_TYPE_F32;
        case DType::bf16: return GGML_TYPE_BF16;
        case DType::i32: return GGML_TYPE_I32;
        default:
            throw Error(ErrorCode::unsupported,
                        "weight dtype is not supported by GGML");
    }
}

} // namespace

WeightStore::WeightStore(ggml_backend_t backend,
                         std::size_t tensor_capacity)
    : backend_(backend) {
    if (backend_ == nullptr || tensor_capacity == 0 ||
        tensor_capacity >
            (std::numeric_limits<std::size_t>::max() /
                 ggml_tensor_overhead()) - 32) {
        throw Error(ErrorCode::invalid_argument,
                    "WeightStore construction arguments are invalid");
    }
    ggml_init_params parameters{};
    parameters.mem_size =
        (tensor_capacity + 32) * ggml_tensor_overhead();
    parameters.no_alloc = true;
    context_ = ggml_init(parameters);
    if (context_ == nullptr) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate GGML weight metadata context");
    }
    tensors_.reserve(tensor_capacity);
}

WeightStore::~WeightStore() {
    if (buffer_ != nullptr) ggml_backend_buffer_free(buffer_);
    if (context_ != nullptr) ggml_free(context_);
}

ggml_tensor * WeightStore::define(
    const std::string & name, DType dtype,
    const std::vector<std::int64_t> & shape) {
    if (buffer_ != nullptr || name.empty() || shape.empty() ||
        shape.size() > GGML_MAX_DIMS ||
        std::any_of(shape.begin(), shape.end(),
                    [](std::int64_t dimension) { return dimension <= 0; }) ||
        tensors_.find(name) != tensors_.end()) {
        throw Error(ErrorCode::failed_precondition,
                    "cannot define GGML weight", {{name, "invalid state"}});
    }
    ggml_tensor * tensor = ggml_new_tensor(
        context_, to_ggml_type(dtype), static_cast<int>(shape.size()),
        shape.data());
    if (tensor == nullptr) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot define GGML weight", {{name, "metadata"}});
    }
    ggml_set_name(tensor, name.c_str());
    tensors_.emplace(name, tensor);
    return tensor;
}

void WeightStore::allocate() {
    if (buffer_ != nullptr || tensors_.empty()) {
        throw Error(ErrorCode::failed_precondition,
                    "WeightStore is not ready for allocation");
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
    if (buffer_ == nullptr) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot allocate GGML weight buffer");
    }
}

void WeightStore::set(const std::string & name, const void * data,
                      std::size_t bytes) {
    ggml_tensor * tensor = find(name);
    if (buffer_ == nullptr || tensor == nullptr || data == nullptr ||
        bytes != ggml_nbytes(tensor)) {
        throw Error(ErrorCode::invalid_argument,
                    "GGML weight upload contract is invalid",
                    {{name, "size, data, or allocation mismatch"}});
    }
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

ggml_tensor * WeightStore::find(const std::string & name) const noexcept {
    const auto item = tensors_.find(name);
    return item == tensors_.end() ? nullptr : item->second;
}

std::uint64_t WeightStore::bytes() const noexcept {
    return buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(buffer_);
}

std::size_t WeightStore::size() const noexcept { return tensors_.size(); }

} // namespace wam::internal::ggml_backend
