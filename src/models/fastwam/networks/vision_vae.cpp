#include "models/fastwam/networks/vision_vae.h"

#include "models/fastwam/networks/ops.h"

#include "backends/ggml/graph_context.h"
#include "backends/ggml/tensor_io.h"
#include "wam/error.h"

#include "ggml-alloc.h"

#include <array>
#include <cmath>
#include <string>

namespace wam::internal::fastwam {
namespace {

ggml_tensor * bias_4d(ggml_context * ctx, ggml_tensor * bias,
                      std::int64_t channels) {
    return ggml_reshape_4d(ctx, bias, 1, 1, channels, 1);
}

ggml_tensor * conv(ggml_context * ctx, ModelResources & resources,
                   const std::string & prefix, ggml_tensor * input,
                   int stride, int padding) {
    ggml_tensor * weight = ops::require_weight(resources, prefix + ".weight");
    ggml_tensor * bias = ops::require_weight(resources, prefix + ".bias");
    if (input->type != GGML_TYPE_BF16) input = ggml_cast(ctx, input, GGML_TYPE_BF16);
    ggml_tensor * output = ggml_conv_2d_direct(
        ctx, weight, input, stride, stride, padding, padding, 1, 1);
    return ggml_add(ctx, output, bias_4d(ctx, bias, output->ne[2]));
}

ggml_tensor * rms(ggml_context * ctx, ModelResources & resources,
                  const std::string & name, ggml_tensor * input) {
    const std::int64_t channels = input->ne[2];
    ggml_tensor * channel_first = ggml_cont(
        ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
    ggml_tensor * gamma = ggml_reshape_1d(
        ctx, ops::require_weight(resources, name), channels);
    if (input->type == GGML_TYPE_BF16 && gamma->type == GGML_TYPE_BF16) {
        return ggml_cont(ctx, ggml_permute(
            ctx, ggml_rms_norm_bf16_f32(ctx, channel_first, gamma, 1.0e-12f),
            2, 0, 1, 3));
    }
    if (channel_first->type != GGML_TYPE_F32) {
        channel_first = ggml_cast(ctx, channel_first, GGML_TYPE_F32);
    }
    ggml_tensor * normalized = ggml_rms_norm(ctx, channel_first, 1.0e-12f);
    gamma = ggml_repeat_4d(
        ctx, gamma, channels, input->ne[0], input->ne[1], input->ne[3]);
    return ggml_cont(ctx, ggml_permute(
        ctx, ggml_mul(ctx, normalized, ggml_cast(ctx, gamma, GGML_TYPE_F32)),
        2, 0, 1, 3));
}

ggml_tensor * residual(ggml_context * ctx, ModelResources & resources,
                       const std::string & prefix, ggml_tensor * input) {
    ggml_tensor * shortcut = input;
    if (resources.weight((prefix + ".skip.weight").c_str())) {
        shortcut = conv(ctx, resources, prefix + ".skip", input, 1, 0);
    }
    ggml_tensor * hidden = rms(ctx, resources, prefix + ".res.0.gamma", input);
    hidden = ggml_silu(ctx, hidden);
    hidden = conv(ctx, resources, prefix + ".res.2", hidden, 1, 1);
    hidden = rms(ctx, resources, prefix + ".res.3.gamma", hidden);
    hidden = ggml_silu(ctx, hidden);
    hidden = conv(ctx, resources, prefix + ".res.6", hidden, 1, 1);
    return ggml_add(ctx, hidden, shortcut);
}

ggml_tensor * avg_shortcut(ggml_context * ctx, ggml_tensor * input,
                           std::int64_t out_channels, bool temporal,
                           bool spatial) {
    if (!spatial && !temporal) return input;
    const bool restore_bf16 = input->type == GGML_TYPE_BF16;
    if (restore_bf16) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * pooled = spatial
        ? ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG,
                       2, 2, 2, 2, 0.0f, 0.0f)
        : input;
    if (!temporal) {
        return restore_bf16 ? ggml_cast(ctx, pooled, GGML_TYPE_BF16) : pooled;
    }
    const std::int64_t width = pooled->ne[0];
    const std::int64_t height = pooled->ne[1];
    const std::int64_t channels = pooled->ne[2];
    if (out_channels != channels * 2) {
        throw Error(ErrorCode::incompatible_artifact,
                    "FastWAM VAE temporal shortcut shape mismatch");
    }
    ggml_tensor * shaped = ggml_reshape_4d(
        ctx, pooled, width, height, 1, channels);
    ggml_tensor * interleaved = ggml_concat(
        ctx, ggml_scale(ctx, shaped, 0.0f), shaped, 2);
    ggml_tensor * output = ggml_reshape_4d(
        ctx, ggml_cont(ctx, interleaved), width, height, out_channels, 1);
    return restore_bf16 ? ggml_cast(ctx, output, GGML_TYPE_BF16) : output;
}

ggml_tensor * down_block(ggml_context * ctx, ModelResources & resources, int index,
                         ggml_tensor * input, bool downsample,
                         bool temporal, std::int64_t out_channels) {
    const std::string prefix =
        "fastwam.vae.encoder.down." + std::to_string(index);
    ggml_tensor * original = input;
    ggml_tensor * hidden = residual(
        ctx, resources, prefix + ".downsamples.0", input);
    hidden = residual(ctx, resources, prefix + ".downsamples.1", hidden);
    if (downsample) {
        if (hidden->type == GGML_TYPE_BF16) {
            hidden = ggml_cast(ctx, hidden, GGML_TYPE_F32);
        }
        hidden = ggml_pad(ctx, hidden, 1, 1, 0, 0);
        hidden = conv(ctx, resources, prefix + ".downsamples.2.rs.1",
                      hidden, 2, 0);
    }
    ggml_tensor * shortcut = avg_shortcut(
        ctx, original, out_channels, temporal, downsample);
    return ggml_add(ctx, hidden, shortcut);
}

ggml_tensor * middle_attention(ggml_context * ctx, ModelResources & resources,
                               ggml_tensor * input) {
    const std::string prefix = "fastwam.vae.encoder.mid.1";
    const std::int64_t width = input->ne[0];
    const std::int64_t height = input->ne[1];
    const std::int64_t channels = input->ne[2];
    const std::int64_t tokens = width * height;
    ggml_tensor * hidden = rms(ctx, resources, prefix + ".norm.gamma", input);
    hidden = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, hidden, 1, 2, 0, 3)),
        channels, tokens);
    ggml_tensor * qkv_weight = ggml_reshape_2d(
        ctx, ops::require_weight(resources, prefix + ".qkv.weight"),
        channels, 3 * channels);
    ggml_tensor * qkv = ops::linear(
        ctx, qkv_weight, ops::require_weight(resources, prefix + ".qkv.bias"), hidden);
    const std::size_t element_size = ggml_type_size(qkv->type);
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(
        ctx, qkv, channels, tokens, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(
        ctx, qkv, channels, tokens, qkv->nb[1],
        static_cast<std::size_t>(channels) * element_size));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(
        ctx, qkv, channels, tokens, qkv->nb[1],
        static_cast<std::size_t>(2 * channels) * element_size));
    q = ggml_reshape_3d(ctx, q, channels, 1, tokens);
    k = ggml_reshape_3d(ctx, k, channels, 1, tokens);
    v = ggml_reshape_3d(ctx, v, channels, 1, tokens);
    ggml_tensor * attended = ops::attention(
        ctx, q, k, v, channels, 1, tokens);
    ggml_tensor * projection = ggml_reshape_2d(
        ctx, ops::require_weight(resources, prefix + ".p.weight"), channels, channels);
    attended = ops::linear(
        ctx, projection, ops::require_weight(resources, prefix + ".p.bias"), attended);
    attended = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_3d(ctx, attended, channels, width, height),
        2, 0, 1, 3));
    return ggml_add(ctx, input, attended);
}

constexpr std::array<float, 48> kMean = {
    -0.2289f,-0.0052f,-0.1323f,-0.2339f,-0.2799f,0.0174f,0.1838f,0.1557f,
    -0.1382f,0.0542f,0.2813f,0.0891f,0.1570f,-0.0098f,0.0375f,-0.1825f,
    -0.2246f,-0.1207f,-0.0698f,0.5109f,0.2665f,-0.2108f,-0.2158f,0.2502f,
    -0.2055f,-0.0322f,0.1109f,0.1567f,-0.0729f,0.0899f,-0.2799f,-0.1230f,
    -0.0313f,-0.1649f,0.0117f,0.0723f,-0.2839f,-0.2083f,-0.0520f,0.3748f,
    0.0152f,0.1957f,0.1433f,-0.2944f,0.3573f,-0.0548f,-0.1681f,-0.0667f,
};
constexpr std::array<float, 48> kStd = {
    0.4765f,1.0364f,0.4514f,1.1677f,0.5313f,0.4990f,0.4818f,0.5013f,
    0.8158f,1.0344f,0.5894f,1.0901f,0.6885f,0.6165f,0.8454f,0.4978f,
    0.5759f,0.3523f,0.7135f,0.6804f,0.5833f,1.4146f,0.8986f,0.5659f,
    0.7069f,0.5338f,0.4889f,0.4917f,0.4069f,0.4999f,0.6866f,0.4093f,
    0.5709f,0.6065f,0.6415f,0.4944f,0.5726f,1.2042f,0.5458f,1.6887f,
    0.3971f,1.0600f,0.3943f,0.5537f,0.5444f,0.4089f,0.7468f,0.7744f,
};

} // namespace

std::vector<ggml_bf16_t> encode_first_frame(
    ModelResources & resources, const FastWamContract & artifact,
    const std::vector<float> & patchified_pixels) {
    const Geometry & g = artifact.geometry;
    const std::size_t expected_pixels =
        static_cast<std::size_t>(12) * (g.image_height / 2) * (g.image_width / 2);
    if (patchified_pixels.size() != expected_pixels || g.latent_channels != 48) {
        throw Error(ErrorCode::invalid_argument,
                    "FastWAM VAE input shape is invalid");
    }
    ggml_backend::GraphContext graph_context(128u * 1024u * 1024u);
    ggml_context * ctx = graph_context.get();

    ggml_tensor * pixels = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, g.image_width / 2, g.image_height / 2, 12, 1);
    ggml_set_input(pixels);
    ggml_tensor * hidden = conv(
        ctx, resources, "fastwam.vae.encoder.in", pixels, 1, 1);
    hidden = down_block(ctx, resources, 0, hidden, true, false, 160);
    hidden = down_block(ctx, resources, 1, hidden, true, true, 320);
    hidden = down_block(ctx, resources, 2, hidden, true, true, 640);
    hidden = down_block(ctx, resources, 3, hidden, false, false, 640);
    hidden = residual(ctx, resources, "fastwam.vae.encoder.mid.0", hidden);
    hidden = middle_attention(ctx, resources, hidden);
    hidden = residual(ctx, resources, "fastwam.vae.encoder.mid.2", hidden);
    hidden = rms(ctx, resources, "fastwam.vae.encoder.out.0.gamma", hidden);
    hidden = ggml_silu(ctx, hidden);
    hidden = conv(ctx, resources, "fastwam.vae.encoder.out.2", hidden, 1, 1);
    hidden = conv(ctx, resources, "fastwam.vae.quant", hidden, 1, 0);
    ggml_tensor * output = ggml_cast(ctx, hidden, GGML_TYPE_F32);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(graph, output);
    graph_context.allocate(
        graph, ggml_backend_get_default_buffer_type(resources.backend()));
    ggml_backend::set_f32(pixels, patchified_pixels);
    if (ggml_backend_graph_compute(resources.backend(), graph) != GGML_STATUS_SUCCESS) {
        throw Error(ErrorCode::inference_failed,
                    "FastWAM VAE graph execution failed");
    }
    const std::size_t plane =
        static_cast<std::size_t>(g.image_height / 16) * (g.image_width / 16);
    const std::vector<float> moments = ggml_backend::get_f32(output);
    std::vector<ggml_bf16_t> result(static_cast<std::size_t>(48) * plane);
    for (std::size_t channel = 0; channel < 48; ++channel) {
        const float mean = ggml_bf16_to_fp32(
            ggml_fp32_to_bf16(kMean[channel]));
        const float inverse_std = ggml_bf16_to_fp32(
            ggml_fp32_to_bf16(1.0F / kStd[channel]));
        for (std::size_t index = 0; index < plane; ++index) {
            const std::size_t offset = channel * plane + index;
            const float centered = ggml_bf16_to_fp32(ggml_fp32_to_bf16(
                moments[offset] - mean));
            result[offset] = ggml_fp32_to_bf16(centered * inverse_std);
        }
    }
    return result;
}

} // namespace wam::internal::fastwam
