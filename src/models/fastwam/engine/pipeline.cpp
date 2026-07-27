#include "pipeline.h"

#include "action_dit.h"
#include "debug.h"
#include "scheduler.h"
#include "vae.h"
#include "video_dit.h"

#include "ggml.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace wam::internal::fastwam {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - begin).count();
}

std::vector<float> vae_pixels(const policy::CpuImage & image,
                              const Geometry & geometry) {
    if (image.layout != policy::TensorLayout::chw || image.channels != 3 ||
        image.height != geometry.image_height ||
        image.width != geometry.image_width ||
        image.height % 2 != 0 || image.width % 2 != 0) {
        throw Error(ErrorCode::inference_failed,
                    "FastWAM composite image geometry is invalid");
    }
    const std::size_t source_plane =
        static_cast<std::size_t>(image.height) * image.width;
    const std::uint32_t output_height = image.height / 2;
    const std::uint32_t output_width = image.width / 2;
    const std::size_t output_plane =
        static_cast<std::size_t>(output_height) * output_width;
    std::vector<float> result(12 * output_plane);
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        for (std::uint32_t x_offset = 0; x_offset < 2; ++x_offset) {
            for (std::uint32_t y_offset = 0; y_offset < 2; ++y_offset) {
                const std::uint32_t output_channel =
                    (channel * 2 + x_offset) * 2 + y_offset;
                for (std::uint32_t y = 0; y < output_height; ++y) {
                    for (std::uint32_t x = 0; x < output_width; ++x) {
                        const std::size_t source = channel * source_plane +
                            static_cast<std::size_t>(y * 2 + y_offset) *
                                image.width + x * 2 + x_offset;
                        const std::size_t target = output_channel * output_plane +
                            static_cast<std::size_t>(y) * output_width + x;
                        result[target] = image.pixels[source];
                    }
                }
            }
        }
    }
    return result;
}

std::vector<ggml_bf16_t> embedding_bf16(const Tensor & tensor) {
    const std::size_t elements = static_cast<std::size_t>(tensor.shape[0]) *
        static_cast<std::size_t>(tensor.shape[1]);
    std::vector<ggml_bf16_t> result(elements);
    if (tensor.dtype == DType::bf16) {
        std::memcpy(result.data(), tensor.data.data(), tensor.data.size());
    } else {
        ggml_fp32_to_bf16_row(
            reinterpret_cast<const float *>(tensor.data.data()), result.data(),
            static_cast<std::int64_t>(elements));
    }
    return result;
}

void append_proprio(std::vector<ggml_bf16_t> & context,
                    std::vector<std::int32_t> & mask,
                    const std::vector<float> & state,
                    const ArtifactContract & artifact) {
    const Geometry & geometry = artifact.geometry;
    std::vector<float> token(geometry.text_dim);
    for (std::uint32_t output = 0; output < geometry.text_dim; ++output) {
        float value = artifact.proprio_bias[output];
        const std::size_t row =
            static_cast<std::size_t>(output) * geometry.proprio_dim;
        for (std::uint32_t input = 0; input < geometry.proprio_dim; ++input) {
            value += artifact.proprio_weight[row + input] * state[input];
        }
        token[output] = value;
    }
    const std::size_t offset = context.size();
    context.resize(offset + token.size());
    ggml_fp32_to_bf16_row(token.data(), context.data() + offset,
                          static_cast<std::int64_t>(token.size()));
    mask.push_back(1);
}

std::vector<ggml_bf16_t> initial_noise(const PreparedInputs & inputs) {
    const std::vector<float> & noise = inputs.observation.action_noise;
    std::vector<ggml_bf16_t> result(noise.size());
    ggml_fp32_to_bf16_row(noise.data(), result.data(),
                          static_cast<std::int64_t>(result.size()));
    return result;
}

} // namespace

CoreAction run_pipeline(Engine & engine, const ArtifactContract & artifact,
                        const PreparedInputs & inputs) {
    const Geometry & geometry = artifact.geometry;
    CoreAction result;
    const Clock::time_point model_begin = Clock::now();

    std::vector<ggml_bf16_t> context = embedding_bf16(inputs.embedding);
    std::vector<std::int32_t> context_mask =
        inputs.embedding_attention_mask;
    append_proprio(context, context_mask, inputs.observation.model_state,
                   artifact);
    const std::size_t context_tokens = context_mask.size();
    debug::dump("context", context,
                {static_cast<std::int64_t>(context_tokens),
                 static_cast<std::int64_t>(geometry.text_dim)});

    Clock::time_point phase_begin = Clock::now();
    const std::vector<float> pixels =
        vae_pixels(inputs.observation.composite_image, geometry);
    debug::dump("vae_pixels", pixels,
                {12, static_cast<std::int64_t>(geometry.image_height / 2),
                 static_cast<std::int64_t>(geometry.image_width / 2)});
    const std::vector<ggml_bf16_t> latent = encode_first_frame(
        engine, artifact, pixels);
    debug::dump("vae_latent", latent,
                {static_cast<std::int64_t>(geometry.latent_channels),
                 static_cast<std::int64_t>(
                     artifact.sequence_geometry.latent_height),
                 static_cast<std::int64_t>(
                     artifact.sequence_geometry.latent_width)});
    result.stats.model_vision_milliseconds = elapsed_ms(phase_begin);
    result.stats.model_timings.push_back(
        {"observation_encoder", result.stats.model_vision_milliseconds});

    phase_begin = Clock::now();
    const VideoKvCache video_cache = prefill_video_cache(
        engine, artifact, latent, context, context_mask, context_tokens);
    if (debug::enabled()) {
        const std::vector<std::int64_t> cache_shape = {
            static_cast<std::int64_t>(video_cache.tokens),
            static_cast<std::int64_t>(geometry.num_heads),
            static_cast<std::int64_t>(geometry.attn_head_dim)};
        for (std::uint32_t layer = 0; layer < geometry.num_layers; ++layer) {
            const std::string index = layer < 10
                ? "0" + std::to_string(layer) : std::to_string(layer);
            debug::dump("video_k_" + index, video_cache.layers[layer].key,
                        cache_shape);
            debug::dump("video_v_" + index, video_cache.layers[layer].value,
                        cache_shape);
        }
    }
    result.stats.model_prefill_milliseconds = elapsed_ms(phase_begin);
    result.stats.model_timings.push_back(
        {"backbone_prefill", result.stats.model_prefill_milliseconds});

    std::vector<ggml_bf16_t> action = initial_noise(inputs);
    const std::vector<std::int64_t> action_shape = {
        static_cast<std::int64_t>(geometry.action_horizon),
        static_cast<std::int64_t>(geometry.action_dim)};
    debug::dump("action_state_00", action, action_shape);
    const FlowSchedule schedule = make_inference_schedule(
        static_cast<int>(geometry.inference_steps), geometry.action_shift);
    const std::vector<std::int32_t> positions =
        semantics::action_positions(artifact.sequence_geometry);
    phase_begin = Clock::now();
    std::vector<float> action_f32(action.size());
    for (std::size_t step = 0; step < schedule.steps(); ++step) {
        const std::vector<float> velocity = run_action_dit_step(
            engine, artifact, action, context, context_tokens, context_mask,
            video_cache, schedule.timesteps[step], positions);
        const std::string step_name = step + 1 < 10
            ? "0" + std::to_string(step + 1)
            : std::to_string(step + 1);
        debug::dump("action_velocity_" + step_name, velocity, action_shape);
        ggml_bf16_to_fp32_row(action.data(), action_f32.data(),
                              static_cast<std::int64_t>(action.size()));
        for (std::size_t index = 0; index < action_f32.size(); ++index) {
            action_f32[index] += velocity[index] * schedule.deltas[step];
        }
        ggml_fp32_to_bf16_row(action_f32.data(), action.data(),
                              static_cast<std::int64_t>(action.size()));
        debug::dump("action_state_" + step_name, action, action_shape);
    }
    result.stats.model_decode_milliseconds = elapsed_ms(phase_begin);
    result.stats.model_timings.push_back(
        {"action_denoise", result.stats.model_decode_milliseconds});
    ggml_bf16_to_fp32_row(action.data(), action_f32.data(),
                          static_cast<std::int64_t>(action.size()));
    debug::dump("action_normalized", action_f32, action_shape);
    result.values = std::move(action_f32);
    result.stats.model_milliseconds = elapsed_ms(model_begin);
    result.stats.peak_device_memory_bytes = engine.resident_device_bytes;
    return result;
}

} // namespace wam::internal::fastwam
