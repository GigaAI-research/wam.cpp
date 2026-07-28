#include "support/test_utils.h"

#include "policy/tensor_input.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

int main() {
    using wam::test::require;
    using wam::test::require_error;

    const std::vector<float> source = {1.0F, -2.0F};
    std::vector<std::uint8_t> unaligned(sizeof(float) * source.size() + 1);
    std::memcpy(unaligned.data() + 1, source.data(),
                source.size() * sizeof(float));
    wam::TensorView tensor;
    tensor.data = unaligned.data() + 1;
    tensor.byte_size = source.size() * sizeof(float);
    tensor.dtype = wam::DType::f32;
    tensor.shape = {2};
    tensor.byte_order = wam::ByteOrder::little;
    require(wam::internal::policy::copy_f32_tensor(tensor, {{2}}, "value") ==
                source,
            "unaligned F32 tensor copy changed");

    tensor.shape = {1, 2};
    require_error(
        [&] {
            (void) wam::internal::policy::copy_f32_tensor(
                tensor, {{2}}, "value");
        },
        wam::ErrorCode::invalid_argument, "unexpected tensor shape");
    tensor.shape = {2};
    tensor.byte_order = wam::ByteOrder::big;
    require_error(
        [&] {
            (void) wam::internal::policy::copy_f32_tensor(
                tensor, {{2}}, "value");
        },
        wam::ErrorCode::invalid_argument, "wrong tensor byte order");

    std::vector<float> nonfinite = {
        1.0F, std::numeric_limits<float>::infinity()};
    tensor.data = nonfinite.data();
    tensor.byte_order = wam::ByteOrder::little;
    require_error(
        [&] {
            (void) wam::internal::policy::copy_f32_tensor(
                tensor, {{2}}, "value");
        },
        wam::ErrorCode::invalid_argument, "non-finite tensor payload");
    return 0;
}
