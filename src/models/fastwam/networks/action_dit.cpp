#include "models/fastwam/networks/action_dit.h"

#include "models/fastwam/networks/ops.h"

#include "backends/ggml/graph_context.h"
#include "backends/ggml/graph_ops.h"
#include "backends/ggml/tensor_io.h"
#include "wam/error.h"

#include "ggml-alloc.h"

#include <cmath>
#include <limits>
#include <string>

namespace wam::internal::fastwam {
namespace {

using ggml_backend::as_bf16;
using ggml_backend::as_f32;

ggml_tensor * repeat_column(ggml_context * ctx, ggml_tensor * value,
                            std::int64_t hidden, std::int64_t tokens) {
    ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    return ggml_repeat(ctx, value, target);
}

ggml_tensor * block(ggml_context * ctx, ModelResources & resources, int layer,
                    ggml_tensor * input, ggml_tensor * context,
                    ggml_tensor * context_tokens, ggml_tensor * positions,
                    ggml_tensor * video_key, ggml_tensor * video_value,
                    ggml_tensor * context_attention_mask,
                    std::int64_t hidden, std::int64_t action_tokens,
                    std::int64_t heads, std::int64_t head_dim, float eps) {
    const std::string prefix = "fastwam.action.blk." + std::to_string(layer);
    context_tokens = ggml_reshape_2d(ctx, context_tokens, hidden, 6);
    ggml_tensor * modulation = ggml_add(
        ctx, as_f32(ctx, ops::require_weight(resources, prefix + ".mod")), context_tokens);
    ggml_tensor * shift_msa = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1], 0),
        hidden, action_tokens);
    ggml_tensor * scale_msa = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1],
                          modulation->nb[1]), hidden, action_tokens);
    ggml_tensor * gate_msa = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1],
                          2 * modulation->nb[1]), hidden, action_tokens);
    ggml_tensor * shift_mlp = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1],
                          3 * modulation->nb[1]), hidden, action_tokens);
    ggml_tensor * scale_mlp = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1],
                          4 * modulation->nb[1]), hidden, action_tokens);
    ggml_tensor * gate_mlp = repeat_column(
        ctx, ggml_view_2d(ctx, modulation, hidden, 1, modulation->nb[1],
                          5 * modulation->nb[1]), hidden, action_tokens);

    ggml_tensor * normalized = ops::modulated_norm(
        ctx, input, scale_msa, shift_msa, eps);
    ggml_tensor * q = ops::linear(ctx, ops::require_weight(resources, prefix + ".sa.q.weight"),
                                  ops::require_weight(resources, prefix + ".sa.q.bias"), normalized);
    ggml_tensor * k = ops::linear(ctx, ops::require_weight(resources, prefix + ".sa.k.weight"),
                                  ops::require_weight(resources, prefix + ".sa.k.bias"), normalized);
    ggml_tensor * v = ops::linear(ctx, ops::require_weight(resources, prefix + ".sa.v.weight"),
                                  ops::require_weight(resources, prefix + ".sa.v.bias"), normalized);
    q = ops::rms_norm(ctx, q, ops::require_weight(resources, prefix + ".sa.qn.weight"), eps);
    k = ops::rms_norm(ctx, k, ops::require_weight(resources, prefix + ".sa.kn.weight"), eps);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, action_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, action_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, action_tokens);
    q = ops::rope_1d(ctx, q, positions, head_dim);
    k = ops::rope_1d(ctx, k, positions, head_dim);
    k = ggml_concat(ctx, video_key, k, 2);
    v = ggml_concat(ctx, video_value, v, 2);
    ggml_tensor * self = ops::attention(ctx, q, k, v, head_dim, heads, action_tokens);
    self = ops::linear(ctx, ops::require_weight(resources, prefix + ".sa.o.weight"),
                       ops::require_weight(resources, prefix + ".sa.o.bias"), self);
    input = ops::gated_residual(ctx, input, self, gate_msa);

    normalized = ops::layer_norm(
        ctx, input, ops::require_weight(resources, prefix + ".norm3.weight"),
        ops::require_weight(resources, prefix + ".norm3.bias"), eps);
    q = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.q.weight"),
                    ops::require_weight(resources, prefix + ".ca.q.bias"), normalized);
    k = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.k.weight"),
                    ops::require_weight(resources, prefix + ".ca.k.bias"), context);
    v = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.v.weight"),
                    ops::require_weight(resources, prefix + ".ca.v.bias"), context);
    q = ops::rms_norm(ctx, q, ops::require_weight(resources, prefix + ".ca.qn.weight"), eps);
    k = ops::rms_norm(ctx, k, ops::require_weight(resources, prefix + ".ca.kn.weight"), eps);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, action_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, context->ne[1]);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, context->ne[1]);
    ggml_tensor * cross = ops::attention(
        ctx, q, k, v, head_dim, heads, action_tokens,
        context_attention_mask);
    cross = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.o.weight"),
                        ops::require_weight(resources, prefix + ".ca.o.bias"), cross);
    input = ggml_add(ctx, input, cross);

    normalized = ops::modulated_norm(ctx, input, scale_mlp, shift_mlp, eps);
    ggml_tensor * ff = ops::linear_gelu(
        ctx, ops::require_weight(resources, prefix + ".ff.0.weight"),
        ops::require_weight(resources, prefix + ".ff.0.bias"),
        ops::require_weight(resources, prefix + ".ff.2.weight"),
        ops::require_weight(resources, prefix + ".ff.2.bias"), normalized);
    return ops::gated_residual(ctx, input, ff, gate_mlp);
}

std::vector<float> sinusoidal(float timestep, int dimension) {
    std::vector<float> values(static_cast<std::size_t>(dimension));
    const int half = dimension / 2;
    for (int index = 0; index < half; ++index) {
        const double exponent = -std::log(10000.0) * static_cast<double>(index) /
            static_cast<double>(half);
        const double value = static_cast<double>(timestep) * std::exp(exponent);
        values[static_cast<std::size_t>(index)] = static_cast<float>(std::cos(value));
        values[static_cast<std::size_t>(half + index)] = static_cast<float>(std::sin(value));
    }
    return values;
}

std::vector<float> build_context_mask(
    const std::vector<std::int32_t> & valid,
    std::size_t queries, std::size_t heads) {
    std::vector<float> result(valid.size() * queries * heads);
    const float blocked = -std::numeric_limits<float>::infinity();
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t query = 0; query < queries; ++query) {
            for (std::size_t key = 0; key < valid.size(); ++key) {
                result[(head * queries + query) * valid.size() + key] =
                    valid[key] ? 0.0f : blocked;
            }
        }
    }
    return result;
}

} // namespace

std::vector<float> run_action_dit_step(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & action_input,
    const std::vector<ggml_bf16_t> & context,
    std::size_t context_tokens,
    const std::vector<std::int32_t> & context_mask,
    const VideoKvCache & video_cache,
    float timestep,
    const std::vector<std::int32_t> & positions) {
    const Geometry & g = artifact.geometry;
    const std::size_t expected_action = static_cast<std::size_t>(g.action_horizon) * g.action_dim;
    const std::size_t expected_context = context_tokens * g.text_dim;
    const std::size_t cache_elements = video_cache.tokens * g.num_heads * g.attn_head_dim;
    if (action_input.size() != expected_action || context.size() != expected_context ||
        context_mask.size() != context_tokens ||
        positions.size() != g.action_horizon || video_cache.tokens == 0 ||
        video_cache.layers.size() != g.num_layers) {
        throw Error(ErrorCode::invalid_argument, "FastWAM action graph input shape mismatch");
    }
    ggml_backend::GraphContext graph_context(512u * 1024u * 1024u);
    ggml_context * ctx = graph_context.get();

    ggml_tensor * action = ggml_new_tensor_2d(
        ctx, GGML_TYPE_BF16, g.action_dim, g.action_horizon);
    ggml_tensor * context_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_BF16, g.text_dim, context_tokens);
    ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * position_input = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, g.action_horizon);
    ggml_tensor * context_attention_mask = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, context_tokens, g.action_horizon, g.num_heads);
    ggml_set_input(action); ggml_set_input(context_input);
    ggml_set_input(freqs); ggml_set_input(position_input);
    ggml_set_input(context_attention_mask);

    ggml_tensor * hidden = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.encoder.weight"),
        ops::require_weight(resources, "fastwam.action.encoder.bias"), action);
    ggml_tensor * text = ops::linear_gelu(
        ctx, ops::require_weight(resources, "fastwam.action.text.0.weight"),
        ops::require_weight(resources, "fastwam.action.text.0.bias"),
        ops::require_weight(resources, "fastwam.action.text.2.weight"),
        ops::require_weight(resources, "fastwam.action.text.2.bias"), context_input);

    ggml_tensor * t = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.time.0.weight"),
        ops::require_weight(resources, "fastwam.action.time.0.bias"), freqs);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, ops::require_weight(resources, "fastwam.action.time.2.weight"),
                    ops::require_weight(resources, "fastwam.action.time.2.bias"), t);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, ops::require_weight(resources, "fastwam.action.timep.1.weight"),
                    ops::require_weight(resources, "fastwam.action.timep.1.bias"), t);
    t = as_f32(ctx, t);
    std::vector<ggml_tensor *> video_keys;
    std::vector<ggml_tensor *> video_values;
    video_keys.reserve(g.num_layers);
    video_values.reserve(g.num_layers);
    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        if (video_cache.layers[layer].key.size() != cache_elements ||
            video_cache.layers[layer].value.size() != cache_elements) {
            throw Error(ErrorCode::invalid_argument,
                        "FastWAM video K/V cache layer shape mismatch",
                        {{"layer", std::to_string(layer)}});
        }
        ggml_tensor * video_key = ggml_new_tensor_3d(
            ctx, GGML_TYPE_BF16, g.attn_head_dim, g.num_heads, video_cache.tokens);
        ggml_tensor * video_value = ggml_new_tensor_3d(
            ctx, GGML_TYPE_BF16, g.attn_head_dim, g.num_heads, video_cache.tokens);
        ggml_set_input(video_key); ggml_set_input(video_value);
        video_keys.push_back(video_key); video_values.push_back(video_value);
        hidden = block(ctx, resources, static_cast<int>(layer), hidden, text, t,
                       position_input, video_key, video_value,
                       context_attention_mask,
                       g.action_hidden_dim, g.action_horizon,
                       g.num_heads, g.attn_head_dim, g.norm_eps);
    }
    ggml_tensor * output = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.head.weight"),
        ops::require_weight(resources, "fastwam.action.head.bias"), hidden);
    output = as_f32(ctx, output);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, output);
    graph_context.allocate(
        graph, ggml_backend_get_default_buffer_type(resources.backend()));
    ggml_backend::set_bf16(action, action_input);
    ggml_backend::set_bf16(context_input, context);
    const std::vector<float> frequency = sinusoidal(timestep, 256);
    ggml_backend::set_f32(freqs, frequency);
    ggml_backend_tensor_set(position_input, positions.data(), 0,
                            positions.size() * sizeof(std::int32_t));
    const std::vector<float> mask = build_context_mask(
        context_mask, g.action_horizon, g.num_heads);
    ggml_backend::set_f32(context_attention_mask, mask);
    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        ggml_backend::set_bf16(video_keys[layer],
                               video_cache.layers[layer].key);
        ggml_backend::set_bf16(video_values[layer],
                               video_cache.layers[layer].value);
    }
    if (ggml_backend_graph_compute(resources.backend(), graph) != GGML_STATUS_SUCCESS) {
        throw Error(ErrorCode::inference_failed, "FastWAM ActionDiT graph execution failed");
    }
    return ggml_backend::get_f32(output);
}

} // namespace wam::internal::fastwam
