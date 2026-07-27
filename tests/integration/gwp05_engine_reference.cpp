#include "support/test_utils.h"

#include "models/common/gguf_reader.h"
#include "models/gwp05/artifact.h"
#include "models/gwp05/engine/engine.h"
#include "models/gwp05/inputs.h"
#include "wam/wam.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct Image {
    std::vector<std::uint8_t> pixels;
    int width = 0;
    int height = 0;
};

Image read_ppm(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    std::string magic;
    int maximum = 0;
    Image image;
    if (!(input >> magic >> image.width >> image.height >> maximum) ||
        magic != "P6" || image.width <= 0 || image.height <= 0 ||
        maximum != 255) {
        throw std::runtime_error("invalid PPM input: " + path.string());
    }
    input.get();
    image.pixels.resize(
        static_cast<std::size_t>(image.width) * image.height * 3U);
    if (!input.read(reinterpret_cast<char *>(image.pixels.data()),
                    static_cast<std::streamsize>(image.pixels.size()))) {
        throw std::runtime_error("truncated PPM input: " + path.string());
    }
    return image;
}

std::vector<float> read_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("missing F32 input: " + path.string());
    }
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 input size: " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(
        static_cast<std::size_t>(bytes) / sizeof(float));
    if (!input.read(reinterpret_cast<char *>(values.data()), bytes)) {
        throw std::runtime_error("truncated F32 input: " + path.string());
    }
    return values;
}

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         std::string layout) {
    return {values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), std::move(layout), wam::ByteOrder::little};
}

void compare_stage(const std::filesystem::path & actual_root,
                   const std::filesystem::path & donor_root,
                   const std::string & name, bool exact,
                   double mean_tolerance, double max_tolerance) {
    const std::vector<float> actual =
        read_f32(actual_root / (name + ".f32"));
    const std::vector<float> expected =
        read_f32(donor_root / (name + ".f32"));
    wam::test::require(actual.size() == expected.size(),
                       name + ": element count differs");
    if (exact) {
        wam::test::require(
            std::memcmp(actual.data(), expected.data(),
                        actual.size() * sizeof(float)) == 0,
            name + ": exact scheduler boundary differs");
        return;
    }
    double absolute_sum = 0.0;
    double absolute_max = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = std::fabs(
            static_cast<double>(actual[index]) - expected[index]);
        absolute_sum += difference;
        absolute_max = std::max(absolute_max, difference);
    }
    const double mean = absolute_sum / actual.size();
    if (mean > mean_tolerance || absolute_max > max_tolerance) {
        throw std::runtime_error(
            name + ": mean_abs=" + std::to_string(mean) +
            " max_abs=" + std::to_string(absolute_max));
    }
}

} // namespace

int main(int argc, char ** argv) {
    using wam::internal::GgufReader;
    namespace gwp05 = wam::internal::gwp05;
    namespace engine = wam::internal::gwp05::engine;

    if (argc != 4) {
        throw std::runtime_error(
            "usage: wam_gwp05_engine_reference MODEL_GGUF INPUT_DIR "
            "DONOR_STAGE_DIR");
    }
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path input_root = argv[2];
    const std::filesystem::path donor_root = argv[3];
    const std::filesystem::path dump_root =
        std::filesystem::temp_directory_path() /
        ("wam-gwp05-slice4b-" + std::to_string(getpid()));
    std::filesystem::create_directories(dump_root);
    if (setenv("WAM_GWP05_DUMP_DIR", dump_root.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set GWP dump directory");
    }

    Image high = read_ppm(input_root / "camera_high.ppm");
    Image left = read_ppm(input_root / "camera_left_wrist.ppm");
    Image right = read_ppm(input_root / "camera_right_wrist.ppm");
    std::vector<float> state = read_f32(input_root / "state.f32");
    std::vector<float> noise = read_f32(input_root / "noise.f32");
    std::vector<float> embedding =
        read_f32(input_root / "t5_embedding.f32");
    wam::test::require(state.size() == 14 && noise.size() == 48 * 32 &&
                           embedding.size() == 64 * 4096,
                       "frozen replay tensor sizes changed");

    auto reader = GgufReader::open(model_path.string());
    const wam::internal::policy::PolicySpecDraft policy_spec =
        gwp05::read_legacy_policy_spec(*reader);
    const std::shared_ptr<const gwp05::ArtifactContract> artifact =
        gwp05::load_artifact(reader, policy_spec);

    const auto image_view = [](const char * name, const Image & image) {
        return wam::ImageView{
            name, wam::ImageEncoding::rgb_u8, image.pixels.data(),
            image.pixels.size(), static_cast<std::uint32_t>(image.width),
            static_cast<std::uint32_t>(image.height), 3,
            static_cast<std::size_t>(image.width) * 3U};
    };
    std::vector<std::int32_t> embedding_mask(64, 1);
    wam::Inputs inputs;
    inputs.images = {
        image_view("camera_high", high),
        image_view("camera_left_wrist", left),
        image_view("camera_right_wrist", right),
    };
    inputs.state = f32_view(state, {14}, "D");
    inputs.action_noise = f32_view(noise, {48, 32}, "T,A");
    inputs.language = wam::EmbeddingInput{
        f32_view(embedding, {64, 4096}, "T,D"),
        {embedding_mask.data(), embedding_mask.size() * sizeof(std::int32_t),
         wam::DType::i32, {64}, "T", wam::ByteOrder::little}};
    std::mt19937 session_rng(20260713);
    const gwp05::PreparedInputs prepared = gwp05::prepare_inputs(
        inputs, *artifact, policy_spec,
        wam::LanguageRuntimeMode::external_embedding, session_rng);

    engine::EngineOptions options;
    options.backend = wam::Backend::automatic;
    options.compute_precision = wam::ComputePrecision::f32;
    options.language_mode = wam::LanguageRuntimeMode::external_embedding;
    options.prompt_cache_capacity = 0;
    engine::EnginePtr engine_model =
        engine::create_engine(*artifact, options);
    engine::EngineSessionOptions session_options;
    engine::EngineSessionPtr engine_session =
        engine::create_engine_session(*engine_model, session_options);
    const gwp05::CoreAction action =
        engine::predict(*engine_session, prepared);
    wam::test::require(action.values.size() == 48 * 32,
                       "core action shape changed");

    const std::vector<std::string> intermediate = {
        "image_input", "normalized_state", "vae_conv_in", "vae_down0",
        "vae_down1", "vae_down2", "vae_down3", "vae_latent",
        "t5_embedding", "action_tokens", "visual_tokens",
        "action_condition", "block0_action", "velocity_step0"};
    for (const std::string & stage : intermediate) {
        compare_stage(dump_root, donor_root, stage, false, 2.0e-4, 1.0e-2);
    }
    compare_stage(dump_root, donor_root, "flow_timesteps", true, 0.0, 0.0);
    compare_stage(dump_root, donor_root, "flow_sigmas", true, 0.0, 0.0);
    for (int step = 0; step < 10; ++step) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "denoise_%02d", step);
        compare_stage(dump_root, donor_root,
                      std::string(prefix) + "_action_in", false,
                      1.0e-3, 1.0e-2);
        compare_stage(dump_root, donor_root,
                      std::string(prefix) + "_dt", true, 0.0, 0.0);
        compare_stage(dump_root, donor_root,
                      std::string(prefix) + "_velocity", false,
                      step == 9 ? 5.0e-3 : 2.0e-4, 5.0e-2);
        compare_stage(dump_root, donor_root,
                      std::string(prefix) + "_action_out", false,
                      1.0e-3, 1.0e-2);
    }
    const std::vector<float> expected_action =
        read_f32(donor_root / "denoise_09_action_out.f32");
    double final_mean = 0.0;
    double final_max = 0.0;
    for (std::size_t index = 0; index < action.values.size(); ++index) {
        const double difference = std::fabs(
            static_cast<double>(action.values[index]) - expected_action[index]);
        final_mean += difference;
        final_max = std::max(final_max, difference);
    }
    final_mean /= action.values.size();
    wam::test::require(final_mean <= 1.0e-3 && final_max <= 1.0e-3,
                       "returned normalized action differs from donor");

    engine::reset(*engine_session);
    const gwp05::CoreAction repeated =
        engine::predict(*engine_session, prepared);
    double repeat_mean = 0.0;
    double repeat_max = 0.0;
    for (std::size_t index = 0; index < action.values.size(); ++index) {
        const double difference = std::fabs(
            static_cast<double>(repeated.values[index]) - action.values[index]);
        repeat_mean += difference;
        repeat_max = std::max(repeat_max, difference);
    }
    repeat_mean /= action.values.size();
    wam::test::require(repeat_mean <= 1.0e-3 && repeat_max <= 1.0e-3,
                       "reset repeat exceeds the frozen F32 action tolerance");
    wam::test::require(prepared.observation.action_noise == noise,
                       "engine modified explicit action noise");

    engine_session.reset();
    engine_model.reset();

    wam::ModelOptions model_options;
    model_options.artifact_path = model_path.string();
    model_options.backend = wam::Backend::automatic;
    model_options.compute_precision = wam::ComputePrecision::f32;
    model_options.language_mode =
        wam::LanguageRuntimeMode::external_embedding;
    wam::Model * public_model = wam::model_load(model_options);
    const wam::ModelInfo & public_info = wam::model_info(public_model);
    wam::test::require(public_info.capabilities.action &&
                           !public_info.capabilities.token_input &&
                           public_info.capabilities.precomputed_embedding &&
                           public_info.resident_device_bytes > 0 &&
                           public_info.compute_precision ==
                               wam::ComputePrecision::f32,
                       "public GWP model did not expose the Slice 5 runtime");

    wam::SessionOptions public_session_options;
    public_session_options.random_seed = 20260713;
    wam::Session * first_session =
        wam::session_create(public_model, public_session_options);
    wam::Session * second_session =
        wam::session_create(public_model, public_session_options);
    wam::model_free(public_model);

    const auto prediction_values = [](const wam::Prediction & prediction) {
        wam::test::require(
            prediction.action.dtype == wam::DType::f32 &&
                prediction.action.shape ==
                    std::vector<std::int64_t>({48, 14}) &&
                prediction.action.layout == "T,A" &&
                prediction.action.byte_order == wam::ByteOrder::little &&
                prediction.action.data.size() == 48U * 14U * sizeof(float),
            "public GWP action contract changed");
        std::vector<float> values(48U * 14U);
        std::memcpy(values.data(), prediction.action.data.data(),
                    prediction.action.data.size());
        return values;
    };
    const std::vector<float> expected_public_action =
        read_f32(donor_root / "action.f32");
    const std::vector<float> first_values = prediction_values(
        wam::predict(first_session, inputs));
    double public_mean = 0.0;
    double public_max = 0.0;
    wam::test::require(first_values.size() == expected_public_action.size(),
                       "public donor action element count changed");
    for (std::size_t index = 0; index < first_values.size(); ++index) {
        const double difference = std::fabs(
            static_cast<double>(first_values[index]) -
            expected_public_action[index]);
        public_mean += difference;
        public_max = std::max(public_max, difference);
    }
    public_mean /= first_values.size();
    wam::test::require(public_mean <= 1.0e-3 && public_max <= 1.0e-2,
                       "public GWP action differs from donor recovery");

    wam::Inputs random_inputs = inputs;
    random_inputs.action_noise = {};
    const std::vector<float> first_random_values = prediction_values(
        wam::predict(first_session, random_inputs));
    const std::vector<float> second_random_values = prediction_values(
        wam::predict(second_session, random_inputs));
    wam::test::require(second_random_values == first_random_values,
                       "two GWP sessions did not isolate seeded RNG state");

    wam::test::require(static_cast<bool>(wam::session_reset(first_session)),
                       "public GWP session reset failed");
    const std::vector<float> reset_random_values = prediction_values(
        wam::predict(first_session, random_inputs));
    wam::test::require(reset_random_values == first_random_values,
                       "public GWP reset did not restore the session seed");
    wam::session_free(first_session);
    wam::session_free(second_session);

    unsetenv("WAM_GWP05_DUMP_DIR");
    std::filesystem::remove_all(dump_root);
    std::cout << "GWP05 Slice 4B/5 engine and public lifecycle parity: PASS "
                 "reset_mean_abs="
              << repeat_mean << " reset_max_abs=" << repeat_max << '\n';
    return 0;
}
