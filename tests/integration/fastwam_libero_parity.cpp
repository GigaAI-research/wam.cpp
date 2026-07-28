#include "support/test_utils.h"
#include "wam/wam.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_array(const std::filesystem::path & path,
                          std::size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() !=
                      static_cast<std::streamoff>(count * sizeof(T))) {
        throw std::runtime_error("invalid replay array: " + path.string());
    }
    input.seekg(0);
    std::vector<T> result(count);
    if (!input.read(reinterpret_cast<char *>(result.data()),
                    static_cast<std::streamsize>(count * sizeof(T)))) {
        throw std::runtime_error("truncated replay array: " + path.string());
    }
    return result;
}

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         std::string layout) {
    return {values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), std::move(layout), wam::ByteOrder::little};
}

} // namespace

int main(int argc, char ** argv) {
    using wam::test::require;
    if (argc != 3 && argc != 5) {
        throw std::runtime_error(
            "usage: wam_fastwam_libero_parity MODEL_GGUF REPLAY_DIR "
            "[MEAN_TOLERANCE MAX_TOLERANCE]");
    }
    const double mean_tolerance = argc == 5 ? std::stod(argv[3]) : 9.0e-4;
    const double maximum_tolerance = argc == 5 ? std::stod(argv[4]) : 5.0e-3;
    require(std::isfinite(mean_tolerance) && mean_tolerance > 0.0 &&
                std::isfinite(maximum_tolerance) && maximum_tolerance > 0.0,
            "FastWAM parity tolerances must be finite and positive");
    const std::filesystem::path root = argv[2];
    const std::vector<std::uint8_t> scene = read_array<std::uint8_t>(
        root / "camera0.rgb", 224U * 224U * 3U);
    const std::vector<std::uint8_t> wrist = read_array<std::uint8_t>(
        root / "camera1.rgb", 224U * 224U * 3U);
    const std::vector<std::uint16_t> embedding = read_array<std::uint16_t>(
        root / "context.bf16", 128U * 4096U);
    const std::vector<std::int32_t> mask = read_array<std::int32_t>(
        root / "context-mask.i32", 128U);
    const std::vector<float> state =
        read_array<float>(root / "state.f32", 8U);
    const std::vector<float> noise =
        read_array<float>(root / "noise.f32", 32U * 7U);
    const std::vector<float> expected =
        read_array<float>(root / "expected-action.f32", 32U * 7U);
    wam::RuntimeConfig options;
    options.backend = wam::Backend::cuda;
    options.compute_precision = wam::ComputePrecision::bf16;
    options.language_mode = wam::LanguageRuntimeMode::external_embedding;
    wam::Model model = wam::Model::load(argv[1], options);
    wam::SessionConfig session_options;
    session_options.random_seed = 20260726;
    wam::Session session = model.create_session(session_options);

    const auto image = [](const char * name,
                          const std::vector<std::uint8_t> & pixels) {
        return wam::ImageView{name, wam::ImageEncoding::rgb_u8, pixels.data(),
                              pixels.size(), 224, 224, 3, 224U * 3U};
    };
    wam::Observation inputs;
    inputs.images = {image("wrist", wrist), image("scene", scene)};
    inputs.state = f32_view(state, {8}, "D");
    inputs.action_noise = f32_view(noise, {32, 7}, "T,A");
    inputs.language = wam::EmbeddingInput{
        {embedding.data(), embedding.size() * sizeof(std::uint16_t),
         wam::DType::bf16, {128, 4096}, "T,D", wam::ByteOrder::little},
        {mask.data(), mask.size() * sizeof(std::int32_t), wam::DType::i32,
         {128}, "T", wam::ByteOrder::little}};

    const wam::Prediction prediction = session.predict(inputs);
    require(prediction.action.dtype == wam::DType::f32 &&
                prediction.action.shape ==
                    std::vector<std::int64_t>({32, 7}) &&
                prediction.action.layout == "T,A" &&
                prediction.action.data.size() == expected.size() * sizeof(float),
            "FastWAM public action contract changed");
    std::vector<float> actual(expected.size());
    std::memcpy(actual.data(), prediction.action.data.data(),
                prediction.action.data.size());
    if (const char * output_path =
            std::getenv("WAM_FASTWAM_PARITY_OUTPUT")) {
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char *>(actual.data()),
                     static_cast<std::streamsize>(
                         actual.size() * sizeof(float)));
    }
    double sum = 0.0;
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), "FastWAM action is not finite");
        const double error = std::fabs(
            static_cast<double>(actual[index]) - expected[index]);
        sum += error;
        maximum = std::max(maximum, error);
    }
    const double mean = sum / actual.size();
    std::cerr << "FastWAM LIBERO action parity: mean_abs=" << mean
              << " max_abs=" << maximum << std::endl;
    if (mean > mean_tolerance || maximum > maximum_tolerance) {
        throw std::runtime_error(
            "FastWAM LIBERO action parity exceeded tolerance: mean_abs=" +
            std::to_string(mean) + " max_abs=" +
            std::to_string(maximum));
    }
    require(prediction.telemetry.model_vision_milliseconds > 0.0 &&
                prediction.telemetry.model_prefill_milliseconds > 0.0 &&
                prediction.telemetry.model_decode_milliseconds > 0.0 &&
                prediction.telemetry.model_timings.size() == 3,
            "FastWAM phase timing contract changed");

    inputs.action_noise = {};
    const wam::Prediction first_random = session.predict(inputs);
    const wam::Prediction second_random = session.predict(inputs);
    require(first_random.action.data != second_random.action.data,
            "FastWAM session RNG did not advance between predictions");
    wam::Session peer_session = model.create_session(session_options);
    const wam::Prediction peer_random = peer_session.predict(inputs);
    require(first_random.action.data == peer_random.action.data,
            "FastWAM engine sessions do not own independent RNG state");
    session.reset();
    const wam::Prediction reset_random = session.predict(inputs);
    require(first_random.action.data == reset_random.action.data,
            "FastWAM session reset did not restore its random noise sequence");

    return 0;
}
