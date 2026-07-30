#include "models/fastwam/networks/video_dit.h"

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
    ggml_tensor * target = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, tokens);
    return ggml_repeat(ctx, value, target);
}

struct LayerKv {
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
};

LayerKv block(ggml_context * ctx, ModelResources & resources, int layer,
              ggml_tensor *& input, ggml_tensor * context,
              ggml_tensor * t_mod, ggml_tensor * time_positions,
              ggml_tensor * height_positions, ggml_tensor * width_positions,
              ggml_tensor * context_attention_mask,
              const Geometry & g, std::int64_t tokens) {
    const std::string prefix = "fastwam.video.blk." + std::to_string(layer);
    ggml_tensor * modulation = ggml_add(
        ctx, as_f32(ctx, ops::require_weight(resources, prefix + ".mod")), t_mod);
    auto part = [&](int index) {
        return repeat_column(
            ctx, ggml_view_2d(
                ctx, modulation, g.video_hidden_dim, 1, modulation->nb[1],
                static_cast<std::size_t>(index) * modulation->nb[1]),
            g.video_hidden_dim, tokens);
    };
    ggml_tensor * shift_msa = part(0);
    ggml_tensor * scale_msa = part(1);
    ggml_tensor * gate_msa = part(2);
    ggml_tensor * shift_mlp = part(3);
    ggml_tensor * scale_mlp = part(4);
    ggml_tensor * gate_mlp = part(5);

    ggml_tensor * normalized = ops::modulated_norm(
        ctx, input, scale_msa, shift_msa, g.norm_eps);
    ggml_tensor * q = ops::linear(
        ctx, ops::require_weight(resources, prefix + ".sa.q.weight"),
        ops::require_weight(resources, prefix + ".sa.q.bias"), normalized);
    ggml_tensor * k = ops::linear(
        ctx, ops::require_weight(resources, prefix + ".sa.k.weight"),
        ops::require_weight(resources, prefix + ".sa.k.bias"), normalized);
    ggml_tensor * v = ops::linear(
        ctx, ops::require_weight(resources, prefix + ".sa.v.weight"),
        ops::require_weight(resources, prefix + ".sa.v.bias"), normalized);
    q = ops::rms_norm(ctx, q,
                      ops::require_weight(resources, prefix + ".sa.qn.weight"), g.norm_eps);
    k = ops::rms_norm(ctx, k,
                      ops::require_weight(resources, prefix + ".sa.kn.weight"), g.norm_eps);
    q = ggml_reshape_3d(ctx, q, g.attn_head_dim, g.num_heads, tokens);
    k = ggml_reshape_3d(ctx, k, g.attn_head_dim, g.num_heads, tokens);
    v = ggml_reshape_3d(ctx, v, g.attn_head_dim, g.num_heads, tokens);
    q = ops::rope_3d(ctx, q, time_positions, height_positions, width_positions,
                     g.attn_head_dim, g.num_heads, tokens);
    k = ops::rope_3d(ctx, k, time_positions, height_positions, width_positions,
                     g.attn_head_dim, g.num_heads, tokens);
    LayerKv cache{as_bf16(ctx, ggml_cont(ctx, k)),
                  as_bf16(ctx, ggml_cont(ctx, v))};
    ggml_tensor * self = ops::attention(
        ctx, q, cache.key, cache.value,
        g.attn_head_dim, g.num_heads, tokens);
    self = ops::linear(
        ctx, ops::require_weight(resources, prefix + ".sa.o.weight"),
        ops::require_weight(resources, prefix + ".sa.o.bias"), self);
    input = ops::gated_residual(ctx, input, self, gate_msa);

    normalized = ops::layer_norm(
        ctx, input, ops::require_weight(resources, prefix + ".norm3.weight"),
        ops::require_weight(resources, prefix + ".norm3.bias"), g.norm_eps);
    q = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.q.weight"),
                    ops::require_weight(resources, prefix + ".ca.q.bias"), normalized);
    k = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.k.weight"),
                    ops::require_weight(resources, prefix + ".ca.k.bias"), context);
    v = ops::linear(ctx, ops::require_weight(resources, prefix + ".ca.v.weight"),
                    ops::require_weight(resources, prefix + ".ca.v.bias"), context);
    q = ops::rms_norm(ctx, q,
                      ops::require_weight(resources, prefix + ".ca.qn.weight"), g.norm_eps);
    k = ops::rms_norm(ctx, k,
                      ops::require_weight(resources, prefix + ".ca.kn.weight"), g.norm_eps);
    q = ggml_reshape_3d(ctx, q, g.attn_head_dim, g.num_heads, tokens);
    k = ggml_reshape_3d(ctx, k, g.attn_head_dim, g.num_heads, context->ne[1]);
    v = ggml_reshape_3d(ctx, v, g.attn_head_dim, g.num_heads, context->ne[1]);
    ggml_tensor * cross = ops::attention(
        ctx, q, k, v, g.attn_head_dim, g.num_heads, tokens,
        context_attention_mask);
    cross = ops::linear(
        ctx, ops::require_weight(resources, prefix + ".ca.o.weight"),
        ops::require_weight(resources, prefix + ".ca.o.bias"), cross);
    input = ggml_add(ctx, input, cross);

    normalized = ops::modulated_norm(
        ctx, input, scale_mlp, shift_mlp, g.norm_eps);
    ggml_tensor * ff = ops::linear_gelu(
        ctx, ops::require_weight(resources, prefix + ".ff.0.weight"),
        ops::require_weight(resources, prefix + ".ff.0.bias"),
        ops::require_weight(resources, prefix + ".ff.2.weight"),
        ops::require_weight(resources, prefix + ".ff.2.bias"), normalized);
    input = ops::gated_residual(ctx, input, ff, gate_mlp);
    return cache;
}

std::vector<float> timestep_embedding(float timestep, int dimension) {
    std::vector<float> result(static_cast<std::size_t>(dimension));
    const int half = dimension / 2;
    for (int index = 0; index < half; ++index) {
        const double frequency = std::pow(
            10000.0, -static_cast<double>(index) / half);
        const double angle = static_cast<double>(timestep) * frequency;
        result[static_cast<std::size_t>(index)] = static_cast<float>(std::cos(angle));
        result[static_cast<std::size_t>(half + index)] = static_cast<float>(std::sin(angle));
    }
    return result;
}

std::vector<float> cross_mask(const std::vector<std::int32_t> & valid,
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

void prefill_video_cache(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & latent,
    const std::vector<ggml_bf16_t> & context,
    const std::vector<std::int32_t> & context_mask,
    std::size_t context_tokens, DeviceVideoKvCache & cache) {
    const Geometry & g = artifact.geometry;
    const semantics::SequenceGeometry & sequence = artifact.sequence_geometry;
    const std::size_t expected_latent =
        static_cast<std::size_t>(g.latent_channels) *
        sequence.latent_height * sequence.latent_width;
    if (latent.size() != expected_latent ||
        context.size() != context_tokens * g.text_dim ||
        context_mask.size() != context_tokens || context_tokens == 0) {
        throw Error(ErrorCode::invalid_argument,
                    "FastWAM Video Expert input shape mismatch");
    }

    ggml_backend::GraphContext graph_context(512u * 1024u * 1024u);
    ggml_context * ctx = graph_context.get();
    const std::int64_t tokens = sequence.video_tokens;
    cache.initialize(resources.backend(), g.num_layers,
                     static_cast<std::size_t>(tokens), g.num_heads,
                     g.attn_head_dim);
    ggml_tensor * latent_input = ggml_new_tensor_3d(
        ctx, GGML_TYPE_BF16, sequence.latent_width,
        sequence.latent_height, g.latent_channels);
    ggml_tensor * context_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_BF16, g.text_dim, context_tokens);
    ggml_tensor * frequencies = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * time_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * height_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * width_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * attention_mask = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, context_tokens, tokens, g.num_heads);
    for (ggml_tensor * input : {latent_input, context_input, frequencies,
                                time_positions, height_positions, width_positions,
                                attention_mask}) {
        ggml_set_input(input);
    }

    ggml_tensor * hidden = ggml_conv_2d_direct(
        ctx, ops::require_weight(resources, "fastwam.video.patch.weight"), latent_input,
        2, 2, 0, 0, 1, 1);
    hidden = ggml_add(
        ctx, hidden, ggml_reshape_4d(
            ctx, ops::require_weight(resources, "fastwam.video.patch.bias"),
            1, 1, g.video_hidden_dim, 1));
    hidden = as_bf16(ctx, hidden);
    hidden = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, hidden, 1, 2, 0, 3)),
        g.video_hidden_dim, tokens);

    ggml_tensor * projected_context = ops::linear_gelu(
        ctx, ops::require_weight(resources, "fastwam.video.text.0.weight"),
        ops::require_weight(resources, "fastwam.video.text.0.bias"),
        ops::require_weight(resources, "fastwam.video.text.2.weight"),
        ops::require_weight(resources, "fastwam.video.text.2.bias"), context_input);
    ggml_tensor * t = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.video.time.0.weight"),
        ops::require_weight(resources, "fastwam.video.time.0.bias"), frequencies);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, ops::require_weight(resources, "fastwam.video.time.2.weight"),
                    ops::require_weight(resources, "fastwam.video.time.2.bias"), t);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, ops::require_weight(resources, "fastwam.video.timep.1.weight"),
                    ops::require_weight(resources, "fastwam.video.timep.1.bias"), t);
    ggml_tensor * t_mod = ggml_reshape_2d(
        ctx, as_f32(ctx, t), g.video_hidden_dim, 6);

    std::vector<LayerKv> outputs;
    outputs.reserve(g.num_layers);
    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        outputs.push_back(block(
            ctx, resources, static_cast<int>(layer), hidden, projected_context,
            t_mod, time_positions, height_positions, width_positions,
            attention_mask, g, tokens));
    }
    for (const LayerKv & output : outputs) {
        ggml_set_output(output.key);
        ggml_set_output(output.value);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 131072, false);
    for (const LayerKv & output : outputs) {
        ggml_build_forward_expand(graph, output.key);
        ggml_build_forward_expand(graph, output.value);
    }
    graph_context.allocate(
        graph, ggml_backend_get_default_buffer_type(resources.backend()));

    ggml_backend::set_bf16(latent_input, latent);
    ggml_backend::set_bf16(context_input, context);
    const std::vector<float> frequency = timestep_embedding(0.0f, 256);
    ggml_backend::set_f32(frequencies, frequency);
    const semantics::VisualPositions positions =
        semantics::visual_positions(sequence);
    ggml_backend_tensor_set(time_positions, positions.time.data(), 0,
                            positions.time.size() * sizeof(std::int32_t));
    ggml_backend_tensor_set(height_positions, positions.height.data(), 0,
                            positions.height.size() * sizeof(std::int32_t));
    ggml_backend_tensor_set(width_positions, positions.width.data(), 0,
                            positions.width.size() * sizeof(std::int32_t));
    const std::vector<float> mask = cross_mask(
        context_mask, static_cast<std::size_t>(tokens), g.num_heads);
    ggml_backend::set_f32(attention_mask, mask);
    if (ggml_backend_graph_compute(resources.backend(), graph) != GGML_STATUS_SUCCESS) {
        throw Error(ErrorCode::inference_failed,
                    "FastWAM Video Expert graph execution failed");
    }

    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        ggml_backend_tensor_copy(outputs[layer].key, cache.key(layer));
        ggml_backend_tensor_copy(outputs[layer].value, cache.value(layer));
    }
    ggml_backend_synchronize(resources.backend());
}

} // namespace wam::internal::fastwam
