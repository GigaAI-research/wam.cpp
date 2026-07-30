#include "models/fastwam/pipeline.h"

#include "models/fastwam/networks/action_dit.h"
#include "models/fastwam/networks/proprio_projector.h"
#include "models/fastwam/scheduler.h"
#include "models/fastwam/state.h"
#include "models/fastwam/networks/vision_vae.h"
#include "models/fastwam/networks/video_dit.h"

#include "wam/error.h"
#include "runtime/telemetry.h"

#include "ggml.h"

#include <chrono>
#include <cmath>
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
                        const float pixel = image.pixels[source];
                        const float u8 = std::round((pixel + 1.0F) * 127.5F);
                        const ggml_bf16_t scaled = ggml_fp32_to_bf16(
                            u8 * (2.0F / 255.0F));
                        result[target] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(
                            ggml_bf16_to_fp32(scaled) - 1.0F));
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
                    const std::vector<ggml_bf16_t> & token) {
    const std::size_t offset = context.size();
    context.resize(offset + token.size());
    std::copy(token.begin(), token.end(), context.begin() + offset);
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

LoadedModel load_model_resources(const FastWamContract & contract,
                                 const ModelOptions & options) {
    if (options.backend != Backend::automatic &&
        options.backend != Backend::cuda) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM execution requires the CUDA backend");
    }
    if (options.compute_precision != ComputePrecision::automatic &&
        options.compute_precision != ComputePrecision::bf16) {
        throw Error(ErrorCode::unsupported,
                    "FastWAM supports BF16 compute only");
    }
    auto logger = options.logger
        ? options.logger
        : std::make_shared<runtime::Logger>(RuntimeConfig{});
    auto debug_dump = options.debug_dump
        ? options.debug_dump
        : std::make_shared<ggml_backend::DebugDump>();
    auto resources = std::make_shared<ModelResources>(
        std::move(logger), std::move(debug_dump));
    if (!resources->initialize(contract, options.device_index)) {
        throw Error(ErrorCode::resource_exhausted,
                    "cannot initialize FastWAM BF16 CUDA resources");
    }
    runtime::EngineInfo info;
    info.backend = Backend::cuda;
    info.compute_precision = ComputePrecision::bf16;
    info.resident_device_bytes = resources->resident_device_bytes;
    info.peak_component_device_bytes = resources->resident_device_bytes;
    info.runtime_components = resources->runtime_components;
    return {std::move(resources), std::move(info)};
}

std::unique_ptr<SessionState> create_session_state() {
    return std::make_unique<SessionState>();
}

CoreAction run_pipeline(ModelResources & resources, SessionState & state,
                        const FastWamContract & contract,
                        const PreparedInputs & inputs) {
    std::lock_guard<std::mutex> lock(resources.execution_mutex);
    const Geometry & geometry = contract.geometry;
    CoreAction result;
    const Clock::time_point model_begin = Clock::now();

    std::vector<ggml_bf16_t> context = embedding_bf16(inputs.embedding);
    std::vector<std::int32_t> context_mask =
        inputs.embedding_attention_mask;
    append_proprio(context, context_mask,
                   project_proprio(inputs.observation.model_state, contract));
    const std::size_t context_tokens = context_mask.size();
    resources.debug_dump().write("context", context,
                {static_cast<std::int64_t>(context_tokens),
                 static_cast<std::int64_t>(geometry.text_dim)});

    Clock::time_point phase_begin = Clock::now();
    const std::vector<float> pixels =
        vae_pixels(inputs.observation.composite_image, geometry);
    resources.debug_dump().write("vae_pixels", pixels,
                {12, static_cast<std::int64_t>(geometry.image_height / 2),
                 static_cast<std::int64_t>(geometry.image_width / 2)});
    const std::vector<ggml_bf16_t> latent = encode_first_frame(
        resources, contract, pixels);
    resources.debug_dump().write("vae_latent", latent,
                {static_cast<std::int64_t>(geometry.latent_channels),
                 static_cast<std::int64_t>(
                     contract.sequence_geometry.latent_height),
                 static_cast<std::int64_t>(
                     contract.sequence_geometry.latent_width)});
    result.stats.model_vision_milliseconds = elapsed_ms(phase_begin);
    runtime::append_timing(result.stats, "observation_encoder",
                           result.stats.model_vision_milliseconds);

    phase_begin = Clock::now();
    DeviceVideoKvCache & video_cache = resources.video_cache();
    prefill_video_cache(resources, contract, latent, context, context_mask,
                        context_tokens, video_cache);
    if (resources.debug_dump().enabled()) {
        for (std::uint32_t layer = 0; layer < geometry.num_layers; ++layer) {
            const std::string index = layer < 10
                ? "0" + std::to_string(layer) : std::to_string(layer);
            resources.debug_dump().write_tensor(
                "video_k_" + index, video_cache.key(layer));
            resources.debug_dump().write_tensor(
                "video_v_" + index, video_cache.value(layer));
        }
    }
    result.stats.model_prefill_milliseconds = elapsed_ms(phase_begin);
    runtime::append_timing(result.stats, "backbone_prefill",
                           result.stats.model_prefill_milliseconds);

    std::vector<ggml_bf16_t> action = initial_noise(inputs);
    const std::vector<std::int64_t> action_shape = {
        static_cast<std::int64_t>(geometry.action_horizon),
        static_cast<std::int64_t>(geometry.action_dim)};
    resources.debug_dump().write("action_state_00", action, action_shape);
    const FlowSchedule schedule = make_inference_schedule(
        static_cast<int>(geometry.inference_steps), geometry.action_shift);
    const std::vector<std::int32_t> positions =
        semantics::action_positions(contract.sequence_geometry);
    phase_begin = Clock::now();
    std::vector<float> action_f32 = run_unrolled_action_denoise(
        resources, contract, action, context, context_tokens, context_mask,
        video_cache, schedule, positions);
    result.stats.model_decode_milliseconds = elapsed_ms(phase_begin);
    runtime::append_timing(result.stats, "action_denoise",
                           result.stats.model_decode_milliseconds);
    resources.debug_dump().write("action_normalized", action_f32, action_shape);
    result.values = std::move(action_f32);
    result.stats.model_milliseconds = elapsed_ms(model_begin);
    result.stats.peak_device_memory_bytes = resources.resident_device_bytes +
        resources.execution_device_bytes();
    ++state.prediction_count;
    return result;
}

void reset_session(ModelResources & resources, SessionState & state) {
    std::lock_guard<std::mutex> lock(resources.execution_mutex);
    state.reset();
}

} // namespace wam::internal::fastwam
