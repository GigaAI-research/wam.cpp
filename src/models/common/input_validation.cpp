#include "models/common/input_validation.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace wam::internal {

std::size_t checked_numel(const std::vector<std::int64_t> & shape,
                          const std::string & field) {
    if (shape.empty()) {
        throw Error(ErrorCode::invalid_argument, "tensor shape is empty",
                    {{field, "expected at least one dimension"}});
    }
    std::size_t elements = 1;
    for (std::int64_t dimension : shape) {
        if (dimension <= 0 ||
            elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(dimension)) {
            throw Error(ErrorCode::invalid_argument, "tensor shape is invalid",
                        {{field, "zero, negative, or overflow"}});
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    return elements;
}

std::vector<float> copy_f32_tensor(
    const TensorView & tensor,
    const std::vector<std::vector<std::int64_t>> & allowed_shapes,
    const std::string & field) {
    bool shape_valid = false;
    for (const std::vector<std::int64_t> & shape : allowed_shapes) {
        shape_valid = shape_valid || tensor.shape == shape;
    }
    const std::size_t elements = checked_numel(tensor.shape, field);
    if (tensor.data == nullptr || tensor.dtype != DType::f32 ||
        tensor.byte_order != ByteOrder::little || !shape_valid ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        tensor.byte_size != elements * sizeof(float)) {
        throw Error(ErrorCode::invalid_argument, "tensor contract is invalid",
                    {{field, "expected contiguous little-endian F32"}});
    }
    std::vector<float> result(elements);
    std::memcpy(result.data(), tensor.data, tensor.byte_size);
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!std::isfinite(result[index])) {
            throw Error(ErrorCode::invalid_argument,
                        "tensor contains NaN or Inf",
                        {{field, std::to_string(index)}});
        }
    }
    return result;
}

std::vector<std::int32_t> copy_i32_tensor(
    const TensorView & tensor, const std::vector<std::int64_t> & shape,
    const std::string & field) {
    const std::size_t elements = checked_numel(tensor.shape, field);
    if (tensor.data == nullptr || tensor.dtype != DType::i32 ||
        tensor.byte_order != ByteOrder::little || tensor.shape != shape ||
        elements > std::numeric_limits<std::size_t>::max() /
                       sizeof(std::int32_t) ||
        tensor.byte_size != elements * sizeof(std::int32_t)) {
        throw Error(ErrorCode::invalid_argument, "tensor contract is invalid",
                    {{field, "expected contiguous little-endian I32"}});
    }
    std::vector<std::int32_t> result(elements);
    std::memcpy(result.data(), tensor.data, tensor.byte_size);
    return result;
}

} // namespace wam::internal
