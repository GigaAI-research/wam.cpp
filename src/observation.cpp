#include "wam/observation.h"

namespace wam {

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

} // namespace wam
