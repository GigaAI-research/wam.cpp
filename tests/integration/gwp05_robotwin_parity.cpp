#include "models/common/gguf_reader.h"
#include "models/gwp05/artifact.h"
#include "models/gwp05/inputs.h"
#include "policy/policy_spec.h"
#include "support/test_utils.h"
#include "wam/wam.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Image {
    std::vector<std::uint8_t> pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ErrorMetrics {
    double mean_absolute = 0.0;
    double maximum_absolute = 0.0;
    std::size_t maximum_index = 0;
};

Image read_ppm(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    std::string magic;
    int width = 0;
    int height = 0;
    int maximum = 0;
    if (!(input >> magic >> width >> height >> maximum) || magic != "P6" ||
        width <= 0 || height <= 0 || maximum != 255) {
        throw std::runtime_error("invalid PPM input: " + path.string());
    }
    input.get();
    Image image;
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    image.pixels.resize(
        static_cast<std::size_t>(width) * height * 3U);
    if (!input.read(reinterpret_cast<char *>(image.pixels.data()),
                    static_cast<std::streamsize>(image.pixels.size()))) {
        throw std::runtime_error("truncated PPM input: " + path.string());
    }
    return image;
}

template <typename T>
std::vector<T> read_array(const std::filesystem::path & path,
                          std::size_t expected_elements) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("missing array: " + path.string());
    }
    const std::streamoff bytes = input.tellg();
    const std::size_t expected_bytes = expected_elements * sizeof(T);
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) != expected_bytes) {
        throw std::runtime_error("array size differs: " + path.string());
    }
    input.seekg(0);
    std::vector<T> values(expected_elements);
    if (!input.read(reinterpret_cast<char *>(values.data()), bytes)) {
        throw std::runtime_error("truncated array: " + path.string());
    }
    return values;
}

ErrorMetrics compare_values(const std::vector<float> & actual,
                            const std::vector<float> & expected,
                            const std::string & name,
                            double mean_limit, double maximum_limit) {
    wam::test::require(actual.size() == expected.size(),
                       name + ": element count differs");
    ErrorMetrics metrics;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        wam::test::require(std::isfinite(actual[index]) &&
                               std::isfinite(expected[index]),
                           name + ": NaN or Inf");
        const double error = std::fabs(
            static_cast<double>(actual[index]) - expected[index]);
        metrics.mean_absolute += error;
        if (error > metrics.maximum_absolute) {
            metrics.maximum_absolute = error;
            metrics.maximum_index = index;
        }
    }
    metrics.mean_absolute /= actual.size();
    if (metrics.mean_absolute > mean_limit ||
        metrics.maximum_absolute > maximum_limit) {
        throw std::runtime_error(
            name + ": mean_abs=" + std::to_string(metrics.mean_absolute) +
            " max_abs=" + std::to_string(metrics.maximum_absolute) +
            " max_index=" + std::to_string(metrics.maximum_index) +
            " actual=" + std::to_string(actual[metrics.maximum_index]) +
            " expected=" + std::to_string(expected[metrics.maximum_index]) +
            " limits=" + std::to_string(mean_limit) + "/" +
            std::to_string(maximum_limit));
    }
    return metrics;
}

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         std::string layout) {
    return {values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), std::move(layout), wam::ByteOrder::little};
}

wam::ImageView image_view(const char * name, const Image & image) {
    return {name, wam::ImageEncoding::rgb_u8, image.pixels.data(),
            image.pixels.size(), image.width, image.height, 3,
            static_cast<std::size_t>(image.width) * 3U};
}

std::vector<float> prediction_values(const wam::Prediction & prediction) {
    wam::test::require(
        prediction.action.dtype == wam::DType::f32 &&
            prediction.action.shape ==
                std::vector<std::int64_t>({48, 14}) &&
            prediction.action.layout == "T,A" &&
            prediction.action.byte_order == wam::ByteOrder::little &&
            prediction.action.data.size() == 48U * 14U * sizeof(float),
        "public RoboTwin action contract changed");
    std::vector<float> values(48U * 14U);
    std::memcpy(values.data(), prediction.action.data.data(),
                prediction.action.data.size());
    return values;
}

} // namespace

int main(int argc, char ** argv) {
    namespace gwp05 = wam::internal::gwp05;
    namespace policy = wam::internal::policy;
    using wam::test::require;

    if (argc != 4) {
        throw std::runtime_error(
            "usage: wam_gwp05_robotwin_parity MODEL_GGUF INPUT_DIR "
            "PYTORCH_REFERENCE_DIR");
    }
    const std::filesystem::path model_path =
        std::filesystem::absolute(argv[1]);
    const std::filesystem::path input_root =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path reference_root =
        std::filesystem::absolute(argv[3]);

    const Image high = read_ppm(input_root / "camera_high.ppm");
    const Image left = read_ppm(input_root / "camera_left_wrist.ppm");
    const Image right = read_ppm(input_root / "camera_right_wrist.ppm");
    const std::vector<float> state =
        read_array<float>(input_root / "state.f32", 14);
    const std::vector<float> noise =
        read_array<float>(input_root / "noise.f32", 48U * 14U);
    const std::vector<float> embedding = read_array<float>(
        reference_root / "prompt_embedding.f32", 64U * 4096U);
    const std::vector<std::int32_t> attention_mask =
        read_array<std::int32_t>(
            reference_root / "attention_mask.i32", 64);
    const std::vector<float> expected_image = read_array<float>(
        reference_root / "processed_image.f32", 3U * 384U * 320U);
    const std::vector<float> expected_state = read_array<float>(
        reference_root / "normalized_state.f32", 14);
    const std::vector<float> expected_normalized_action = read_array<float>(
        reference_root / "normalized_action.f32", 48U * 14U);
    const std::vector<float> expected_action = read_array<float>(
        reference_root / "action.f32", 48U * 14U);
    (void) read_array<float>(
        reference_root / "vae_latent.f32", 48U * 24U * 20U);

    wam::Observation inputs;
    inputs.images = {
        image_view("camera_high", high),
        image_view("camera_left_wrist", left),
        image_view("camera_right_wrist", right),
    };
    inputs.state = f32_view(state, {14}, "D");
    inputs.action_noise = f32_view(noise, {48, 14}, "T,A");
    inputs.language = wam::EmbeddingInput{
        f32_view(embedding, {64, 4096}, "T,D"),
        {attention_mask.data(), attention_mask.size() * sizeof(std::int32_t),
         wam::DType::i32, {64}, "T", wam::ByteOrder::little}};

    policy::PolicySpec spec;
    ErrorMetrics image_metrics;
    ErrorMetrics state_metrics;
    {
        std::shared_ptr<wam::internal::GgufReader> reader =
            wam::internal::GgufReader::open(model_path.string());
        const std::optional<policy::PolicySpec> parsed =
            policy::try_read_policy_spec(*reader);
        require(parsed.has_value(), "formal GGUF has no PolicySpec");
        spec = *parsed;
        const std::shared_ptr<const gwp05::ArtifactContract> artifact =
            gwp05::load_artifact(reader, spec);
        std::mt19937 rng(20260725);
        const gwp05::PreparedInputs prepared = gwp05::prepare_inputs(
            inputs, *artifact, spec,
            wam::LanguageRuntimeMode::external_embedding, rng);
        image_metrics = compare_values(
            prepared.observation.composite_image.pixels, expected_image,
            "PolicySpec image preprocessing", 1.0e-2, 3.0e-2);
        state_metrics = compare_values(
            prepared.observation.model_state, expected_state,
            "PolicySpec state z-score", 1.0e-6, 1.0e-5);
        require(prepared.embedding.dtype == wam::DType::f32 &&
                    prepared.embedding.data.size() ==
                        embedding.size() * sizeof(float) &&
                    std::memcmp(prepared.embedding.data.data(),
                                embedding.data(),
                                prepared.embedding.data.size()) == 0,
                "external prompt embedding changed during preprocessing");
        require(prepared.embedding_attention_mask == attention_mask,
                "external prompt attention mask changed");
        require(prepared.observation.action_noise == noise,
                "explicit action noise changed during preprocessing");
    }

    wam::RuntimeConfig options;
    options.backend = wam::Backend::cuda;
    options.compute_precision = wam::ComputePrecision::bf16;
    options.language_mode = wam::LanguageRuntimeMode::external_embedding;
    options.prompt_cache_capacity = 0;
    wam::Model model = wam::Model::load(model_path.string(), options);
    const wam::ModelInfo & info = model.info();
    require(info.backend == wam::Backend::cuda &&
                info.compute_precision == wam::ComputePrecision::bf16 &&
                info.policy_spec != nullptr &&
                info.policy_spec->identity.profile ==
                    "gwp05_robotwin_dual_arm_14d_zscore" &&
                info.capabilities.action &&
                info.capabilities.raw_images &&
                info.capabilities.precomputed_embedding &&
                info.capabilities.explicit_action_noise &&
                info.resident_device_bytes > 0,
            "public model did not select the Gate A BF16/CUDA runtime");

    wam::SessionConfig session_options;
    session_options.enable_prefix_cache = true;
    session_options.random_seed = 20260725;
    wam::Session session = model.create_session(session_options);
    const wam::Prediction prediction = session.predict(inputs);
    const std::vector<float> action = prediction_values(prediction);
    require(prediction.telemetry.model_milliseconds > 0.0 &&
                prediction.telemetry.model_vision_milliseconds > 0.0 &&
                prediction.telemetry.model_decode_milliseconds > 0.0 &&
                prediction.telemetry.total_milliseconds >=
                    prediction.telemetry.model_milliseconds,
            "public BF16 runtime timing phases are incomplete");

    std::vector<float> recovered_normalized(action.size());
    for (std::size_t step = 0; step < spec.action.horizon; ++step) {
        for (std::size_t dimension = 0;
             dimension < spec.action.real_dim; ++dimension) {
            float delta = action[step * spec.action.real_dim + dimension];
            const std::int32_t state_index =
                spec.action.recovery.reference_state_indices[dimension];
            if (state_index >= 0) {
                delta -= state[static_cast<std::size_t>(state_index)];
            }
            recovered_normalized[step * spec.action.model_dim + dimension] =
                (delta - spec.action.stats.mean[dimension]) /
                spec.action.stats.stddev[dimension];
        }
    }
    const ErrorMetrics normalized_metrics = compare_values(
        recovered_normalized, expected_normalized_action,
        "BF16 normalized action", 1.2e-2, 1.0e-1);
    const ErrorMetrics action_metrics = compare_values(
        action, expected_action, "RoboTwin recovered action", 6.0e-3, 4.0e-2);
    const bool strict_pytorch_target =
        action_metrics.mean_absolute <= 1.0e-3 &&
        action_metrics.maximum_absolute <= 1.0e-2;

    session.reset();
    const std::vector<float> repeated = prediction_values(
        session.predict(inputs));
    require(repeated == action,
            "explicit-noise prediction changed after session reset");
    require(noise == read_array<float>(
                input_root / "noise.f32", 48U * 14U),
            "public prediction modified the explicit-noise fixture");

    std::cout << "GWP05 formal RoboTwin BF16/CUDA parity: PASS "
              << "image_mean_abs=" << image_metrics.mean_absolute
              << " image_max_abs=" << image_metrics.maximum_absolute
              << " state_mean_abs=" << state_metrics.mean_absolute
              << " state_max_abs=" << state_metrics.maximum_absolute
              << " normalized_mean_abs=" << normalized_metrics.mean_absolute
              << " normalized_max_abs=" << normalized_metrics.maximum_absolute
              << " action_mean_abs=" << action_metrics.mean_absolute
              << " action_max_abs=" << action_metrics.maximum_absolute
              << " strict_pytorch_target="
              << (strict_pytorch_target ? "PASS" : "FAIL")
              << '\n';
    return 0;
}
