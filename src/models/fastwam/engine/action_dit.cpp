#include "action_dit.h"

#include "ops.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace wam::internal::fastwam {
namespace {

ggml_tensor * as_f32(ggml_context * ctx, ggml_tensor * value) {
    return value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32);
}

ggml_tensor * as_bf16(ggml_context * ctx, ggml_tensor * value) {
    return value->type == GGML_TYPE_BF16 ? value : ggml_cast(ctx, value, GGML_TYPE_BF16);
}

ggml_tensor * require_weight(Engine & engine, const std::string & name) {
    ggml_tensor * value = engine.weight(name.c_str());
    if (!value) {
        throw Error(ErrorCode::incompatible_artifact,
                    "FastWAM action graph weight is missing", {{name, "missing"}});
    }
    return value;
}

ggml_tensor * repeat_column(ggml_context * ctx, ggml_tensor * value,
                            std::int64_t hidden, std::int64_t tokens) {
    ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    return ggml_repeat(ctx, value, target);
}

ggml_tensor * block(ggml_context * ctx, Engine & engine, int layer,
                    ggml_tensor * input, ggml_tensor * context,
                    ggml_tensor * context_tokens, ggml_tensor * positions,
                    ggml_tensor * video_key, ggml_tensor * video_value,
                    ggml_tensor * context_attention_mask,
                    std::int64_t hidden, std::int64_t action_tokens,
                    std::int64_t heads, std::int64_t head_dim, float eps) {
    const std::string prefix = "fastwam.action.blk." + std::to_string(layer);
    context_tokens = ggml_reshape_2d(ctx, context_tokens, hidden, 6);
    ggml_tensor * modulation = ggml_add(
        ctx, as_f32(ctx, require_weight(engine, prefix + ".mod")), context_tokens);
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
    ggml_tensor * q = ops::linear(ctx, require_weight(engine, prefix + ".sa.q.weight"),
                                  require_weight(engine, prefix + ".sa.q.bias"), normalized);
    ggml_tensor * k = ops::linear(ctx, require_weight(engine, prefix + ".sa.k.weight"),
                                  require_weight(engine, prefix + ".sa.k.bias"), normalized);
    ggml_tensor * v = ops::linear(ctx, require_weight(engine, prefix + ".sa.v.weight"),
                                  require_weight(engine, prefix + ".sa.v.bias"), normalized);
    q = ops::rms_norm(ctx, q, require_weight(engine, prefix + ".sa.qn.weight"), eps);
    k = ops::rms_norm(ctx, k, require_weight(engine, prefix + ".sa.kn.weight"), eps);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, action_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, action_tokens);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, action_tokens);
    q = ops::rope_1d(ctx, q, positions, head_dim);
    k = ops::rope_1d(ctx, k, positions, head_dim);
    k = ggml_concat(ctx, video_key, k, 2);
    v = ggml_concat(ctx, video_value, v, 2);
    ggml_tensor * self = ops::attention(ctx, q, k, v, head_dim, heads, action_tokens);
    self = ops::linear(ctx, require_weight(engine, prefix + ".sa.o.weight"),
                       require_weight(engine, prefix + ".sa.o.bias"), self);
    input = ops::gated_residual(ctx, input, self, gate_msa);

    normalized = ops::layer_norm(
        ctx, input, require_weight(engine, prefix + ".norm3.weight"),
        require_weight(engine, prefix + ".norm3.bias"), eps);
    q = ops::linear(ctx, require_weight(engine, prefix + ".ca.q.weight"),
                    require_weight(engine, prefix + ".ca.q.bias"), normalized);
    k = ops::linear(ctx, require_weight(engine, prefix + ".ca.k.weight"),
                    require_weight(engine, prefix + ".ca.k.bias"), context);
    v = ops::linear(ctx, require_weight(engine, prefix + ".ca.v.weight"),
                    require_weight(engine, prefix + ".ca.v.bias"), context);
    q = ops::rms_norm(ctx, q, require_weight(engine, prefix + ".ca.qn.weight"), eps);
    k = ops::rms_norm(ctx, k, require_weight(engine, prefix + ".ca.kn.weight"), eps);
    q = ggml_reshape_3d(ctx, q, head_dim, heads, action_tokens);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, context->ne[1]);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, context->ne[1]);
    ggml_tensor * cross = ops::attention(
        ctx, q, k, v, head_dim, heads, action_tokens,
        context_attention_mask);
    cross = ops::linear(ctx, require_weight(engine, prefix + ".ca.o.weight"),
                        require_weight(engine, prefix + ".ca.o.bias"), cross);
    input = ggml_add(ctx, input, cross);

    normalized = ops::modulated_norm(ctx, input, scale_mlp, shift_mlp, eps);
    ggml_tensor * ff = ops::linear_gelu(
        ctx, require_weight(engine, prefix + ".ff.0.weight"),
        require_weight(engine, prefix + ".ff.0.bias"),
        require_weight(engine, prefix + ".ff.2.weight"),
        require_weight(engine, prefix + ".ff.2.bias"), normalized);
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
    Engine & engine, const ArtifactContract & artifact,
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
    ggml_init_params params{};
    params.mem_size = 512u * 1024u * 1024u;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw Error(ErrorCode::resource_exhausted, "cannot create FastWAM action graph context");

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
        ctx, require_weight(engine, "fastwam.action.encoder.weight"),
        require_weight(engine, "fastwam.action.encoder.bias"), action);
    ggml_tensor * text = ops::linear_gelu(
        ctx, require_weight(engine, "fastwam.action.text.0.weight"),
        require_weight(engine, "fastwam.action.text.0.bias"),
        require_weight(engine, "fastwam.action.text.2.weight"),
        require_weight(engine, "fastwam.action.text.2.bias"), context_input);

    ggml_tensor * t = ops::linear(
        ctx, require_weight(engine, "fastwam.action.time.0.weight"),
        require_weight(engine, "fastwam.action.time.0.bias"), freqs);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, require_weight(engine, "fastwam.action.time.2.weight"),
                    require_weight(engine, "fastwam.action.time.2.bias"), t);
    t = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, t)));
    t = ops::linear(ctx, require_weight(engine, "fastwam.action.timep.1.weight"),
                    require_weight(engine, "fastwam.action.timep.1.bias"), t);
    t = as_f32(ctx, t);
    std::vector<ggml_tensor *> video_keys;
    std::vector<ggml_tensor *> video_values;
    video_keys.reserve(g.num_layers);
    video_values.reserve(g.num_layers);
    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        if (video_cache.layers[layer].key.size() != cache_elements ||
            video_cache.layers[layer].value.size() != cache_elements) {
            ggml_free(ctx);
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
        hidden = block(ctx, engine, static_cast<int>(layer), hidden, text, t,
                       position_input, video_key, video_value,
                       context_attention_mask,
                       g.action_hidden_dim, g.action_horizon,
                       g.num_heads, g.attn_head_dim, g.norm_eps);
    }
    ggml_tensor * output = ops::linear(
        ctx, require_weight(engine, "fastwam.action.head.weight"),
        require_weight(engine, "fastwam.action.head.bias"), hidden);
    output = as_f32(ctx, output);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(engine.backend()));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        throw Error(ErrorCode::resource_exhausted, "cannot allocate FastWAM action graph");
    }
    ggml_backend_tensor_set(action, action_input.data(), 0,
                            action_input.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(context_input, context.data(), 0,
                            context.size() * sizeof(ggml_bf16_t));
    const std::vector<float> frequency = sinusoidal(timestep, 256);
    ggml_backend_tensor_set(freqs, frequency.data(), 0, frequency.size() * sizeof(float));
    ggml_backend_tensor_set(position_input, positions.data(), 0,
                            positions.size() * sizeof(std::int32_t));
    const std::vector<float> mask = build_context_mask(
        context_mask, g.action_horizon, g.num_heads);
    ggml_backend_tensor_set(context_attention_mask, mask.data(), 0,
                            mask.size() * sizeof(float));
    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        ggml_backend_tensor_set(video_keys[layer], video_cache.layers[layer].key.data(), 0,
                                cache_elements * sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(video_values[layer], video_cache.layers[layer].value.data(), 0,
                                cache_elements * sizeof(ggml_bf16_t));
    }
    if (ggml_backend_graph_compute(engine.backend(), graph) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(allocator); ggml_free(ctx);
        throw Error(ErrorCode::inference_failed, "FastWAM ActionDiT graph execution failed");
    }
    std::vector<float> result(expected_action);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return result;
}

} // namespace wam::internal::fastwam
