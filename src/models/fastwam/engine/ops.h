#pragma once

#include "ggml.h"

#include <cstdint>

namespace wam::internal::fastwam::ops {

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight,
                     ggml_tensor * bias, ggml_tensor * input);
ggml_tensor * linear_gelu(ggml_context * ctx, ggml_tensor * in_weight,
                          ggml_tensor * in_bias, ggml_tensor * out_weight,
                          ggml_tensor * out_bias, ggml_tensor * input);
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps);
ggml_tensor * modulated_norm(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * scale, ggml_tensor * shift,
                             float eps);
ggml_tensor * gated_residual(ggml_context * ctx, ggml_tensor * hidden,
                             ggml_tensor * branch, ggml_tensor * gate);
ggml_tensor * rope_1d(ggml_context * ctx, ggml_tensor * input,
                      ggml_tensor * positions, std::int64_t head_dim);
ggml_tensor * rope_3d(ggml_context * ctx, ggml_tensor * input,
                      ggml_tensor * time_positions,
                      ggml_tensor * height_positions,
                      ggml_tensor * width_positions,
                      std::int64_t head_dim, std::int64_t heads,
                      std::int64_t tokens);
ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q,
                        ggml_tensor * k, ggml_tensor * v,
                        std::int64_t head_dim, std::int64_t heads,
                        std::int64_t query_tokens,
                        ggml_tensor * mask = nullptr);

} // namespace wam::internal::fastwam::ops
