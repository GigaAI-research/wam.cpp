#include "support/test_utils.h"

#include "wam/wam.h"

#include <cstdint>
#include <vector>

int main() {
    using wam::test::require;

    const wam::Error error(wam::ErrorCode::invalid_argument, "bad option",
                           {{"field", "reason"}});
    require(error.code() == wam::ErrorCode::invalid_argument,
            "Error must retain its code");
    require(error.details().size() == 1,
            "Error must retain structured details");
    require(error.details().front().field == "field",
            "Error detail field changed");

    require(wam::dtype_size(wam::DType::unknown) == 0,
            "unknown dtype must have no known size");
    require(wam::dtype_size(wam::DType::u8) == 1, "u8 size changed");
    require(wam::dtype_size(wam::DType::bf16) == 2, "bf16 size changed");
    require(wam::dtype_size(wam::DType::i32) == 4, "i32 size changed");
    require(wam::dtype_size(wam::DType::f32) == 4, "f32 size changed");

    std::vector<std::int32_t> values{1, 2, 3};
    const wam::ArrayView<std::int32_t> vector_view(values);
    require(!vector_view.empty() && vector_view.size == values.size(),
            "ArrayView must borrow vector storage");
    require(vector_view.data == values.data(),
            "ArrayView must not copy vector storage");

    const wam::ArrayView<std::int32_t> empty_view;
    require(empty_view.empty(), "default ArrayView must be empty");

    const std::uint8_t byte = 7;
    wam::TensorView tensor_view;
    require(tensor_view.empty(), "default TensorView must be empty");
    tensor_view.data = &byte;
    tensor_view.byte_size = 1;
    require(!tensor_view.empty(), "non-empty TensorView reported empty");

    wam::Tensor tensor;
    require(tensor.empty(), "default Tensor must be empty");
    tensor.data.push_back(byte);
    require(!tensor.empty(), "Tensor with storage reported empty");

    wam::PolicyActionChunk action(std::move(tensor));
    action.shape = {1, 1};
    require(action.horizon() == 1 && action.action_dimension() == 1,
            "PolicyActionChunk shape semantics changed");
    return 0;
}
