#include "input_validation.h"

#include <cmath>
#include <limits>

namespace wam::internal {

std::size_t checked_numel(const std::vector<std::int64_t> & shape,
                          const std::string & field) {
    std::size_t elements = 1;
    for (std::int64_t dimension : shape) {
        if (dimension <= 0 || elements > std::numeric_limits<std::size_t>::max() /
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
    for (const auto & shape : allowed_shapes) shape_valid = shape_valid || tensor.shape == shape;
    const std::size_t elements = checked_numel(tensor.shape, field);
    if (tensor.data == nullptr || tensor.dtype != DType::f32 ||
        tensor.byte_order != ByteOrder::little || !shape_valid ||
        tensor.byte_size != elements * sizeof(float)) {
        throw Error(ErrorCode::invalid_argument, "tensor contract is invalid",
                    {{field, "expected little-endian contiguous F32"}});
    }
    const auto * source = static_cast<const float *>(tensor.data);
    std::vector<float> result(source, source + elements);
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!std::isfinite(result[index])) {
            throw Error(ErrorCode::invalid_argument, "tensor contains NaN or Inf",
                        {{field, std::to_string(index)}});
        }
    }
    return result;
}

} // namespace wam::internal
