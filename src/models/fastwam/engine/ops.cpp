#include "ops.h"

#include "backends/ggml/graph_ops.h"

#include <cmath>

namespace wam::internal::fastwam::ops {
namespace {

using ggml_backend::as_bf16;
using ggml_backend::as_f32;

} // namespace

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight,
                     ggml_tensor * bias, ggml_tensor * input) {
    input = as_bf16(ctx, input);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    output = as_f32(ctx, output);
    if (bias) output = ggml_add(ctx, output, as_f32(ctx, bias));
    return as_bf16(ctx, output);
}

ggml_tensor * linear_gelu(ggml_context * ctx, ggml_tensor * in_weight,
                          ggml_tensor * in_bias, ggml_tensor * out_weight,
                          ggml_tensor * out_bias, ggml_tensor * input) {
    ggml_tensor * hidden = linear(ctx, in_weight, in_bias, input);
    // PyTorch uses approximate="tanh". The pinned ggml GELU implementation
    // uses the same tanh approximation used by the existing GWP backend.
    hidden = as_bf16(ctx, ggml_gelu(ctx, as_f32(ctx, hidden)));
    return linear(ctx, out_weight, out_bias, hidden);
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps) {
    ggml_tensor * output = ggml_rms_norm(ctx, as_f32(ctx, input), eps);
    if (weight) output = ggml_mul(ctx, output, as_f32(ctx, weight));
    return as_bf16(ctx, output);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps) {
    ggml_tensor * output = ggml_norm(ctx, as_f32(ctx, input), eps);
    if (weight) output = ggml_mul(ctx, output, as_f32(ctx, weight));
    if (bias) output = ggml_add(ctx, output, as_f32(ctx, bias));
    return as_bf16(ctx, output);
}

ggml_tensor * modulated_norm(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * scale, ggml_tensor * shift,
                             float eps) {
    ggml_tensor * normalized = ggml_norm(ctx, as_f32(ctx, input), eps);
    scale = as_f32(ctx, scale);
    shift = as_f32(ctx, shift);
    normalized = ggml_add(ctx, normalized, ggml_mul(ctx, normalized, scale));
    return as_bf16(ctx, ggml_add(ctx, normalized, shift));
}

ggml_tensor * gated_residual(ggml_context * ctx, ggml_tensor * hidden,
                             ggml_tensor * branch, ggml_tensor * gate) {
    hidden = as_f32(ctx, hidden);
    branch = as_f32(ctx, branch);
    gate = as_f32(ctx, gate);
    return as_bf16(ctx, ggml_add(ctx, hidden, ggml_mul(ctx, branch, gate)));
}

ggml_tensor * rope_1d(ggml_context * ctx, ggml_tensor * input,
                      ggml_tensor * positions, std::int64_t head_dim) {
    input = as_f32(ctx, input);
    return as_bf16(ctx, ggml_rope_ext(
        ctx, input, positions, nullptr, static_cast<int>(head_dim),
        GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
}

ggml_tensor * rope_3d(ggml_context * ctx, ggml_tensor * input,
                      ggml_tensor * time_positions,
                      ggml_tensor * height_positions,
                      ggml_tensor * width_positions,
                      std::int64_t head_dim, std::int64_t heads,
                      std::int64_t tokens) {
    input = as_f32(ctx, input);
    const std::int64_t spatial = head_dim / 3;
    const std::int64_t sections[3] = {
        head_dim - 2 * spatial, spatial, spatial,
    };
    ggml_tensor * positions[3] = {
        time_positions, height_positions, width_positions,
    };
    ggml_tensor * result = nullptr;
    std::size_t offset = 0;
    for (int section = 0; section < 3; ++section) {
        ggml_tensor * view = ggml_view_3d(
            ctx, input, sections[section], heads, tokens,
            input->nb[1], input->nb[2], offset);
        ggml_tensor * rotated = ggml_rope_ext(
            ctx, view, positions[section], nullptr,
            static_cast<int>(sections[section]), GGML_ROPE_TYPE_NORMAL, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        result = result ? ggml_concat(ctx, result, rotated, 0) : rotated;
        offset += static_cast<std::size_t>(sections[section]) * input->nb[0];
    }
    return as_bf16(ctx, ggml_cont(ctx, result));
}

ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q,
                        ggml_tensor * k, ggml_tensor * v,
                        std::int64_t head_dim, std::int64_t heads,
                        std::int64_t query_tokens, ggml_tensor * mask) {
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = as_f32(ctx, scores);
    ggml_tensor * probabilities = ggml_soft_max_ext(
        ctx, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
    ggml_tensor * attended = ggml_mul_mat(ctx, V, probabilities);
    ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
    attended = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)),
        head_dim * heads, query_tokens);
    return as_bf16(ctx, attended);
}

} // namespace wam::internal::fastwam::ops
