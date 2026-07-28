#include "wam/prediction.h"

#include <utility>

namespace wam {

bool Tensor::empty() const noexcept {
    return data.empty();
}

PolicyActionChunk::PolicyActionChunk(Tensor tensor)
    : Tensor(std::move(tensor)) {}

PolicyActionChunk & PolicyActionChunk::operator=(Tensor tensor) {
    static_cast<Tensor &>(*this) = std::move(tensor);
    return *this;
}

std::size_t PolicyActionChunk::horizon() const noexcept {
    return shape.size() == 2 && shape[0] > 0
        ? static_cast<std::size_t>(shape[0])
        : 0;
}

std::size_t PolicyActionChunk::action_dimension() const noexcept {
    return shape.size() == 2 && shape[1] > 0
        ? static_cast<std::size_t>(shape[1])
        : 0;
}

} // namespace wam
