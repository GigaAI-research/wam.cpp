#include "models/gwp05/networks/vision_vae.h"
#include "models/gwp05/runtime.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace wam::internal::gwp05 {

std::vector<float> patchify_composite(const PipelineInputsView & in,
                                      const ModelGeometry & cfg) {
    const int width = static_cast<int>(cfg.image_width);
    const int height = static_cast<int>(cfg.image_height);
    const int expected = width * height * 3;
    if (!in.composite_image || in.composite_image_n != expected) {
        return {};
    }

    const int patch = 2;
    const int out_width = width / patch;
    const int out_height = height / patch;
    std::vector<float> output(static_cast<size_t>(12) * out_width * out_height);
    for (int c = 0; c < 3; ++c) {
        for (int dx = 0; dx < patch; ++dx) {
            for (int dy = 0; dy < patch; ++dy) {
                const int channel = (c * patch + dx) * patch + dy;
                for (int y = 0; y < out_height; ++y) {
                    for (int x = 0; x < out_width; ++x) {
                        const size_t source =
                            (static_cast<size_t>(c) * height +
                             y * patch + dy) * width + x * patch + dx;
                        const size_t target =
                            (static_cast<size_t>(channel) * out_height + y) * out_width + x;
                        output[target] = in.composite_image[source];
                    }
                }
            }
        }
    }
    return output;
}

ggml_tensor * vae_bias(ggml_context * ctx, ggml_tensor * bias, int64_t channels) {
    return ggml_reshape_4d(ctx, bias, 1, 1, channels, 1);
}

ggml_tensor * vae_conv(ggml_context * ctx, ExecutionState & model,
                       const std::string & prefix, ggml_tensor * input,
                       int stride, int padding) {
    ggml_tensor * weight = model.weight("vae." + prefix + ".weight");
    ggml_tensor * bias = model.weight("vae." + prefix + ".bias");
    ggml_tensor * output = nullptr;
    if (weight->type == GGML_TYPE_BF16) {
        if (input->type != GGML_TYPE_BF16) {
            input = ggml_cast(ctx, input, GGML_TYPE_BF16);
        }
        output = ggml_conv_2d_direct(
            ctx, weight, input, stride, stride, padding, padding, 1, 1);
    } else {
        output = ggml_conv_2d(
            ctx, weight, input, stride, stride, padding, padding, 1, 1);
    }
    return ggml_add(ctx, output, vae_bias(ctx, bias, output->ne[2]));
}

ggml_tensor * vae_rms(ggml_context * ctx, ExecutionState & model,
                      const std::string & name, ggml_tensor * input) {
    const int64_t channels = input->ne[2];
    ggml_tensor * channel_first = ggml_cont(
        ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
    ggml_tensor * gamma = ggml_reshape_1d(ctx, model.weight("vae." + name), channels);
    if (input->type == GGML_TYPE_BF16 && gamma->type == GGML_TYPE_BF16) {
        return ggml_cont(ctx, ggml_permute(
            ctx, ggml_rms_norm_bf16_f32(ctx, channel_first, gamma, 1e-12f),
            2, 0, 1, 3));
    }
    channel_first = ggml_rms_norm(ctx, channel_first, 1e-12f);
    gamma = ggml_repeat_4d(ctx, gamma, channels, input->ne[0], input->ne[1], input->ne[3]);
    channel_first = ggml_mul(ctx, channel_first, gamma);
    return ggml_cont(ctx, ggml_permute(ctx, channel_first, 2, 0, 1, 3));
}

ggml_tensor * vae_residual(ggml_context * ctx, ExecutionState & model,
                           const std::string & prefix, ggml_tensor * input) {
    ggml_tensor * shortcut = input;
    if (model.weights.count(compact_tensor_name("vae." + prefix + ".conv_shortcut.weight"))) {
        shortcut = vae_conv(ctx, model, prefix + ".conv_shortcut", input, 1, 0);
    }
    ggml_tensor * hidden = vae_rms(ctx, model, prefix + ".norm1.gamma", input);
    hidden = ggml_silu(ctx, hidden);
    hidden = vae_conv(ctx, model, prefix + ".conv1", hidden, 1, 1);
    hidden = vae_rms(ctx, model, prefix + ".norm2.gamma", hidden);
    hidden = ggml_silu(ctx, hidden);
    hidden = vae_conv(ctx, model, prefix + ".conv2", hidden, 1, 1);
    return ggml_add(ctx, hidden, shortcut);
}

ggml_tensor * vae_avg_shortcut(ggml_context * ctx, ggml_tensor * input,
                               int64_t out_channels, bool temporal) {
    const bool restore_bf16 = input->type == GGML_TYPE_BF16;
    if (restore_bf16) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * pooled = ggml_pool_2d(
        ctx, input, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    if (!temporal) {
        return restore_bf16 ? ggml_cast(ctx, pooled, GGML_TYPE_BF16) : pooled;
    }
    const int64_t width = pooled->ne[0];
    const int64_t height = pooled->ne[1];
    const int64_t channels = pooled->ne[2];
    if (out_channels != channels * 2) return nullptr;
    ggml_tensor * shaped = ggml_reshape_4d(ctx, pooled, width, height, 1, channels);
    ggml_tensor * zeros = ggml_scale(ctx, shaped, 0.0f);
    ggml_tensor * interleaved = ggml_concat(ctx, zeros, shaped, 2);
    ggml_tensor * shortcut = ggml_reshape_4d(
        ctx, ggml_cont(ctx, interleaved), width, height, out_channels, 1);
    return restore_bf16 ? ggml_cast(ctx, shortcut, GGML_TYPE_BF16) : shortcut;
}

ggml_tensor * vae_down_block(ggml_context * ctx, ExecutionState & model,
                             int index, ggml_tensor * input, bool downsample,
                             bool temporal, int64_t out_channels) {
    const std::string prefix = "encoder.down_blocks." + std::to_string(index);
    ggml_tensor * original = input;
    ggml_tensor * hidden = vae_residual(ctx, model, prefix + ".resnets.0", input);
    hidden = vae_residual(ctx, model, prefix + ".resnets.1", hidden);
    if (!downsample) return ggml_add(ctx, hidden, original);
    if (hidden->type == GGML_TYPE_BF16) {
        // The version-pinned CUDA pad kernel is F32-only. The following
        // convolution still emits BF16 im2col data and uses a BF16 GEMM.
        hidden = ggml_cast(ctx, hidden, GGML_TYPE_F32);
    }
    ggml_tensor * padded = ggml_pad(ctx, hidden, 1, 1, 0, 0);
    hidden = vae_conv(ctx, model, prefix + ".downsampler.resample.1", padded, 2, 0);
    ggml_tensor * shortcut = vae_avg_shortcut(ctx, original, out_channels, temporal);
    return ggml_add(ctx, hidden, shortcut);
}

ggml_tensor * vae_mid_attention(ggml_context * ctx, ExecutionState & model,
                                ggml_tensor * input) {
    const int64_t width = input->ne[0];
    const int64_t height = input->ne[1];
    const int64_t channels = input->ne[2];
    const int64_t tokens = width * height;
    ggml_tensor * hidden = vae_rms(
        ctx, model, "encoder.mid_block.attentions.0.norm.gamma", input);
    hidden = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, hidden, 1, 2, 0, 3)), channels, tokens);
    ggml_tensor * qkv_weight = ggml_reshape_2d(
        ctx, model.weight("vae.encoder.mid_block.attentions.0.to_qkv.weight"),
        channels, 3 * channels);
    ggml_tensor * qkv = linear_with_activation(
        ctx, qkv_weight,
        model.weight("vae.encoder.mid_block.attentions.0.to_qkv.bias"), hidden,
        native_vae_bf16(model));
    const size_t qkv_element_size = ggml_type_size(qkv->type);
    ggml_tensor * q = ggml_view_2d(ctx, qkv, channels, tokens, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(
        ctx, qkv, channels, tokens, qkv->nb[1], channels * qkv_element_size);
    ggml_tensor * v = ggml_view_2d(
        ctx, qkv, channels, tokens, qkv->nb[1], 2 * channels * qkv_element_size);
    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), channels, 1, tokens);
    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), channels, 1, tokens);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), channels, 1, tokens);
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * probabilities = ggml_soft_max_ext(
        ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(channels)), 0.0f);
    ggml_tensor * attended = ggml_mul_mat(ctx, V, probabilities);
    attended = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), channels, tokens);
    ggml_tensor * projection_weight = ggml_reshape_2d(
        ctx, model.weight("vae.encoder.mid_block.attentions.0.proj.weight"),
        channels, channels);
    attended = linear_with_activation(
        ctx, projection_weight,
        model.weight("vae.encoder.mid_block.attentions.0.proj.bias"), attended,
        native_vae_bf16(model));
    attended = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_3d(ctx, attended, channels, width, height), 2, 0, 1, 3));
    return ggml_add(ctx, input, attended);
}

std::vector<float> run_vision_vae(ExecutionState & model, const PipelineInputsView & in) {
    const ModelGeometry & cfg = model.cfg;
    using clock = std::chrono::steady_clock;
    if (!model.vae_graph) model.vae_graph = std::make_unique<VaeGraph>();
    VaeGraph & graph = *model.vae_graph;
    if (!graph.ctx) {
        ggml_init_params params{};
        params.mem_size = 64u * 1024u * 1024u;
        params.no_alloc = true;
        graph.ctx = ggml_init(params);
        if (!graph.ctx) return {};
    }
    ggml_context * ctx = graph.ctx;
    if (!graph.cgraph) {
        graph.pixels = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, cfg.image_width / 2, cfg.image_height / 2, 12, 1);
        ggml_set_input(graph.pixels);
        ggml_tensor * hidden = vae_conv(
            ctx, model, "encoder.conv_in", graph.pixels, 1, 1);
        if (debug_dump_enabled(model)) graph.conv_in_debug = ggml_dup(ctx, hidden);
        hidden = vae_down_block(ctx, model, 0, hidden, true, false, 160);
        if (debug_dump_enabled(model)) graph.down0_debug = ggml_dup(ctx, hidden);
        hidden = vae_down_block(ctx, model, 1, hidden, true, true, 320);
        if (debug_dump_enabled(model)) graph.down1_debug = ggml_dup(ctx, hidden);
        hidden = vae_down_block(ctx, model, 2, hidden, true, true, 640);
        if (debug_dump_enabled(model)) graph.down2_debug = ggml_dup(ctx, hidden);
        hidden = vae_down_block(ctx, model, 3, hidden, false, false, 640);
        if (debug_dump_enabled(model)) graph.down3_debug = ggml_dup(ctx, hidden);
        hidden = vae_residual(ctx, model, "encoder.mid_block.resnets.0", hidden);
        hidden = vae_mid_attention(ctx, model, hidden);
        hidden = vae_residual(ctx, model, "encoder.mid_block.resnets.1", hidden);
        hidden = vae_rms(ctx, model, "encoder.norm_out.gamma", hidden);
        hidden = ggml_silu(ctx, hidden);
        hidden = vae_conv(ctx, model, "encoder.conv_out", hidden, 1, 1);
        hidden = vae_conv(ctx, model, "quant_conv", hidden, 1, 0);
        graph.output = ggml_cast(ctx, hidden, GGML_TYPE_F32);
        ggml_set_output(graph.output);
        graph.cgraph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph.cgraph, graph.output);
        for (ggml_tensor * tensor : {
                 graph.conv_in_debug, graph.down0_debug, graph.down1_debug,
                 graph.down2_debug, graph.down3_debug}) {
            if (tensor) {
                ggml_set_output(tensor);
                ggml_build_forward_expand(graph.cgraph, tensor);
            }
        }
        ggml_graph_assign_uid(graph.cgraph);
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend));
        if (!graph.alloc || !ggml_gallocr_alloc_graph(graph.alloc, graph.cgraph)) return {};
    }
    const auto preprocess_begin = clock::now();
    const std::vector<float> patchified = patchify_composite(in, cfg);
    if (patchified.empty()) return {};
    debug_dump(model, "image_input", patchified,
               {12, cfg.image_height / 2, cfg.image_width / 2});
    ggml_backend_tensor_set(
        graph.pixels, patchified.data(), 0, patchified.size() * sizeof(float));
    model.stats.ms_vae_preprocess = std::chrono::duration<float, std::milli>(
        clock::now() - preprocess_begin).count();
    const auto graph_begin = clock::now();
    if (ggml_backend_graph_compute(model.backend, graph.cgraph) != GGML_STATUS_SUCCESS) return {};
    if (graph.conv_in_debug) debug_dump_tensor(model, "vae_conv_in", graph.conv_in_debug);
    if (graph.down0_debug) debug_dump_tensor(model, "vae_down0", graph.down0_debug);
    if (graph.down1_debug) debug_dump_tensor(model, "vae_down1", graph.down1_debug);
    if (graph.down2_debug) debug_dump_tensor(model, "vae_down2", graph.down2_debug);
    if (graph.down3_debug) debug_dump_tensor(model, "vae_down3", graph.down3_debug);

    const int64_t latent_width = cfg.image_width / 16;
    const int64_t latent_height = cfg.image_height / 16;
    const size_t plane = static_cast<size_t>(latent_width) * latent_height;
    std::vector<float> moments(static_cast<size_t>(2 * cfg.vae_z_dim) * plane);
    ggml_backend_tensor_get(
        graph.output, moments.data(), 0, moments.size() * sizeof(float));
    model.stats.ms_vae_graph = std::chrono::duration<float, std::milli>(
        clock::now() - graph_begin).count();
    std::vector<float> latent(static_cast<size_t>(cfg.vae_z_dim) * plane);
    // Values are the posterior mode (the first z_dim channels), standardized
    // using the Wan2.2 latent statistics embedded in the GGUF metadata.
    // The currently supported config constants are fixed by the converter.
    for (int64_t c = 0; c < cfg.vae_z_dim; ++c) {
        for (size_t i = 0; i < plane; ++i) {
            latent[static_cast<size_t>(c) * plane + i] =
                (moments[static_cast<size_t>(c) * plane + i] - model.vae_latents_mean[c]) /
                model.vae_latents_std[c];
        }
    }
    return latent;
}

} // namespace wam::internal::gwp05
