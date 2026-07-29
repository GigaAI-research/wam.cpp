#include "models/fastwam/networks/proprio_projector.h"

#include "models/fastwam/contract.h"

#include <cstddef>

namespace wam::internal::fastwam {

std::vector<ggml_bf16_t> project_proprio(
    const std::vector<float> & state,
    const FastWamContract & contract) {
    const Geometry & geometry = contract.geometry;
    std::vector<float> projected(geometry.text_dim);
    for (std::uint32_t output = 0; output < geometry.text_dim; ++output) {
        float value = contract.proprio_bias[output];
        const std::size_t row =
            static_cast<std::size_t>(output) * geometry.proprio_dim;
        for (std::uint32_t input = 0; input < geometry.proprio_dim; ++input) {
            value += contract.proprio_weight[row + input] * state[input];
        }
        projected[output] = value;
    }
    std::vector<ggml_bf16_t> token(projected.size());
    ggml_fp32_to_bf16_row(projected.data(), token.data(),
                          static_cast<std::int64_t>(token.size()));
    return token;
}

} // namespace wam::internal::fastwam
