#include "models/fastwam/networks/action_dit.h"

#include "models/fastwam/networks/ops.h"

#include "backends/ggml/graph_context.h"
#include "backends/ggml/graph_ops.h"
#include "backends/ggml/tensor_io.h"
#include "wam/error.h"

#include "ggml-alloc.h"

#include <cmath>
#include <limits>
#include <memory>
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

ggml_tensor * build_action_velocity(
    ggml_context * ctx, ModelResources & resources, const Geometry & g,
    ggml_tensor * action, ggml_tensor * projected_context,
    ggml_tensor * frequencies, ggml_tensor * positions,
    ggml_tensor * context_attention_mask,
    const DeviceVideoKvCache & video_cache) {
    ggml_tensor * hidden = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.encoder.weight"),
        ops::require_weight(resources, "fastwam.action.encoder.bias"), action);
    ggml_tensor * timestep = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.time.0.weight"),
        ops::require_weight(resources, "fastwam.action.time.0.bias"), frequencies);
    timestep = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, timestep)));
    timestep = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.time.2.weight"),
        ops::require_weight(resources, "fastwam.action.time.2.bias"), timestep);
    timestep = as_bf16(ctx, ggml_silu(ctx, as_f32(ctx, timestep)));
    timestep = ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.timep.1.weight"),
        ops::require_weight(resources, "fastwam.action.timep.1.bias"), timestep);
    timestep = as_f32(ctx, timestep);

    for (std::uint32_t layer = 0; layer < g.num_layers; ++layer) {
        hidden = block(
            ctx, resources, static_cast<int>(layer), hidden,
            projected_context, timestep, positions, video_cache.key(layer),
            video_cache.value(layer), context_attention_mask,
            g.action_hidden_dim, g.action_horizon, g.num_heads,
            g.attn_head_dim, g.norm_eps);
    }
    return as_f32(ctx, ops::linear(
        ctx, ops::require_weight(resources, "fastwam.action.head.weight"),
        ops::require_weight(resources, "fastwam.action.head.bias"), hidden));
}

} // namespace

std::vector<float> run_unrolled_action_denoise(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<ggml_bf16_t> & action_input,
    const std::vector<ggml_bf16_t> & context,
    std::size_t context_tokens,
    const std::vector<std::int32_t> & context_mask,
    const DeviceVideoKvCache & video_cache,
    const FlowSchedule & schedule,
    const std::vector<std::int32_t> & positions) {
    const Geometry & g = artifact.geometry;
    const std::size_t expected_action = static_cast<std::size_t>(g.action_horizon) * g.action_dim;
    const std::size_t expected_context = context_tokens * g.text_dim;
    if (action_input.size() != expected_action || context.size() != expected_context ||
        context_mask.size() != context_tokens ||
        positions.size() != g.action_horizon || schedule.steps() == 0 ||
        schedule.deltas.size() != schedule.steps() ||
        !video_cache.matches(g.num_layers,
                             artifact.sequence_geometry.video_tokens,
                             g.num_heads, g.attn_head_dim)) {
        throw Error(ErrorCode::invalid_argument, "FastWAM action graph input shape mismatch");
    }
    const UnrolledActionGraph * existing = resources.action_graph();
    const bool incompatible_graph = existing &&
        (existing->context_tokens != context_tokens ||
         existing->video_tokens != video_cache.tokens() ||
         existing->steps != schedule.steps());
    if (incompatible_graph) resources.reset_action_graph();
    if (resources.action_graph() == nullptr) {
        auto candidate = std::make_unique<UnrolledActionGraph>();
        UnrolledActionGraph & built = *candidate;
        built.context_tokens = context_tokens;
        built.video_tokens = video_cache.tokens();
        built.steps = schedule.steps();
        ggml_context * ctx = built.get();

        built.action_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_BF16, g.action_dim, g.action_horizon);
        const std::size_t action_bytes = ggml_backend_buft_get_alloc_size(
            ggml_backend_get_default_buffer_type(resources.backend()),
            built.action_input);
        built.action_input_buffer =
            ggml_backend_alloc_buffer(resources.backend(), action_bytes);
        if (built.action_input_buffer == nullptr ||
            ggml_backend_tensor_alloc(
                built.action_input_buffer, built.action_input,
                ggml_backend_buffer_get_base(built.action_input_buffer)) !=
                GGML_STATUS_SUCCESS) {
            throw Error(ErrorCode::resource_exhausted,
                        "cannot allocate FastWAM action graph input");
        }
        built.context_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_BF16, g.text_dim, context_tokens);
        built.positions = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, g.action_horizon);
        built.context_attention_mask = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, context_tokens, g.action_horizon,
            g.num_heads);
        for (ggml_tensor * input : {built.context_input, built.positions,
                                    built.context_attention_mask}) {
            ggml_set_input(input);
        }
        ggml_tensor * projected_context = ops::linear_gelu(
            ctx, ops::require_weight(resources, "fastwam.action.text.0.weight"),
            ops::require_weight(resources, "fastwam.action.text.0.bias"),
            ops::require_weight(resources, "fastwam.action.text.2.weight"),
            ops::require_weight(resources, "fastwam.action.text.2.bias"),
            built.context_input);

        const bool keep_debug = resources.debug_dump().enabled();
        ggml_tensor * step_action = built.action_input;
        for (std::size_t step = 0; step < schedule.steps(); ++step) {
            ggml_tensor * frequency =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            ggml_tensor * delta =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            ggml_set_input(frequency);
            ggml_set_input(delta);
            built.frequency_inputs.push_back(frequency);
            built.delta_inputs.push_back(delta);
            ggml_tensor * velocity = build_action_velocity(
                ctx, resources, g, step_action, projected_context, frequency,
                built.positions, built.context_attention_mask, video_cache);
            ggml_tensor * product = as_bf16(
                ctx, ggml_mul(ctx, velocity, delta));
            step_action = as_bf16(
                ctx, ggml_add(ctx, as_f32(ctx, step_action),
                              as_f32(ctx, product)));
            if (keep_debug) {
                ggml_set_output(velocity);
                ggml_set_output(step_action);
                built.debug_velocities.push_back(velocity);
                built.debug_action_states.push_back(step_action);
            }
        }
        built.action_output = as_f32(ctx, step_action);
        ggml_set_output(built.action_output);
        built.graph = ggml_new_graph_custom(ctx, 262144, false);
        ggml_build_forward_expand(built.graph, built.action_output);
        for (std::size_t step = 0; step < built.debug_velocities.size(); ++step) {
            ggml_build_forward_expand(built.graph, built.debug_velocities[step]);
            ggml_build_forward_expand(built.graph, built.debug_action_states[step]);
        }
        ggml_graph_assign_uid(built.graph);
        built.allocate(
            built.graph,
            ggml_backend_get_default_buffer_type(resources.backend()));
        resources.debug_dump().audit_mixed_binary_nodes(
            "fastwam_action_unrolled", built.graph);
        resources.set_action_graph(std::move(candidate));
    }

    UnrolledActionGraph & graph = *resources.action_graph();
    ggml_backend::set_bf16(graph.action_input, action_input);
    ggml_backend::set_bf16(graph.context_input, context);
    ggml_backend_tensor_set(graph.positions, positions.data(), 0,
                            positions.size() * sizeof(std::int32_t));
    const std::vector<float> mask = build_context_mask(
        context_mask, g.action_horizon, g.num_heads);
    ggml_backend::set_f32(graph.context_attention_mask, mask);
    for (std::size_t step = 0; step < schedule.steps(); ++step) {
        ggml_backend::set_f32(
            graph.frequency_inputs[step],
            sinusoidal(schedule.timesteps[step], 256));
        ggml_backend_tensor_set(
            graph.delta_inputs[step], &schedule.deltas[step], 0,
            sizeof(schedule.deltas[step]));
    }
    if (ggml_backend_graph_compute(resources.backend(), graph.graph) !=
        GGML_STATUS_SUCCESS) {
        throw Error(ErrorCode::inference_failed,
                    "FastWAM unrolled ActionDiT graph execution failed");
    }
    for (std::size_t step = 0; step < graph.debug_velocities.size(); ++step) {
        const std::string index = step + 1 < 10
            ? "0" + std::to_string(step + 1) : std::to_string(step + 1);
        resources.debug_dump().write_tensor(
            "action_velocity_" + index, graph.debug_velocities[step]);
        resources.debug_dump().write_tensor(
            "action_state_" + index, graph.debug_action_states[step]);
    }
    return ggml_backend::get_f32(graph.action_output);
}

} // namespace wam::internal::fastwam
