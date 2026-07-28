#include "backends/ggml/tensor_io.h"

#include "wam/error.h"

#include "ggml-backend.h"

namespace wam::internal::ggml_backend {

void set_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor == nullptr ||
        static_cast<std::size_t>(ggml_nelements(tensor)) != values.size()) {
        throw Error(ErrorCode::invalid_argument,
                    "GGML tensor input size mismatch");
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(
            tensor, values.data(), 0, values.size() * sizeof(float));
        return;
    }
    if (tensor->type != GGML_TYPE_BF16) {
        throw Error(ErrorCode::unsupported,
                    "GGML tensor input dtype is unsupported");
    }
    std::vector<ggml_bf16_t> converted(values.size());
    ggml_fp32_to_bf16_row(
        values.data(), converted.data(), converted.size());
    ggml_backend_tensor_set(
        tensor, converted.data(), 0,
        converted.size() * sizeof(ggml_bf16_t));
}

std::vector<float> get_f32(ggml_tensor * tensor) {
    if (tensor == nullptr) {
        throw Error(ErrorCode::invalid_argument, "GGML tensor is null");
    }
    std::vector<float> values(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(
            tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
    if (tensor->type != GGML_TYPE_BF16) {
        throw Error(ErrorCode::unsupported,
                    "GGML tensor output dtype is unsupported");
    }
    std::vector<ggml_bf16_t> raw(values.size());
    ggml_backend_tensor_get(
        tensor, raw.data(), 0, raw.size() * sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row(raw.data(), values.data(), values.size());
    return values;
}

} // namespace wam::internal::ggml_backend
