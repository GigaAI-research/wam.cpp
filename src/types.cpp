#include "wam/types.h"

#include <utility>

namespace wam {

Status::operator bool() const noexcept {
    return code == ErrorCode::ok;
}

Status Status::success() {
    return {};
}

Error::Error(ErrorCode code, std::string message,
             std::vector<ErrorDetail> details)
    : std::runtime_error(std::move(message)),
      code_(code),
      details_(std::move(details)) {}

ErrorCode Error::code() const noexcept {
    return code_;
}

const std::vector<ErrorDetail> & Error::details() const noexcept {
    return details_;
}

std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::u8:
            return 1;
        case DType::i32:
        case DType::f32:
            return 4;
        case DType::bf16:
            return 2;
        case DType::unknown:
            return 0;
    }
    return 0;
}

bool TensorView::empty() const noexcept {
    return data == nullptr || byte_size == 0;
}

bool Tensor::empty() const noexcept {
    return data.empty();
}

} // namespace wam
