#include "models/common/gguf_reader.h"
#include "models/fastwam/artifact.h"
#include "models/fastwam/inputs.h"
#include "policy/policy_spec.h"
#include "support/test_utils.h"
#include "wam/wam.h"

#include <cstdint>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

using wam::internal::GgufReader;
using wam::test::require;

int main(int argc, char ** argv) {
    require(argc == 2, "FastWAM artifact test requires one GGUF path");
    auto reader = GgufReader::open(argv[1]);
    const auto policy_spec =
        wam::internal::policy::try_read_policy_spec_draft(*reader);
    require(policy_spec.has_value(), "FastWAM PolicySpec is missing");
    const auto artifact = wam::internal::fastwam::load_artifact(
        reader, *policy_spec);

    require(policy_spec->identity.profile ==
                "fastwam_libero_2cam224_minmax",
            "FastWAM profile identity changed");
    require(artifact->geometry.image_height == 224 &&
                artifact->geometry.image_width == 448 &&
                artifact->geometry.num_cameras == 2,
            "FastWAM image geometry changed");
    require(artifact->geometry.proprio_dim == 8 &&
                artifact->geometry.action_dim == 7 &&
                artifact->geometry.action_horizon == 32 &&
                std::fabs(artifact->geometry.norm_eps - 1.0e-6F) < 1.0e-12F,
            "FastWAM policy geometry changed");
    require(std::fabs(policy_spec->action.normalization.epsilon - 1.0e-8F) <
                1.0e-14F,
            "FastWAM action normalization epsilon changed");
    require(artifact->components.size() == 5,
            "FastWAM component inventory changed");
    require(reader->tensors().size() == 1741,
            "FastWAM tensor count changed");

    std::vector<std::uint8_t> scene(224U * 224U * 3U, 127);
    std::vector<std::uint8_t> wrist(224U * 224U * 3U, 63);
    std::vector<float> state(8U, 0.0F);
    std::vector<float> embedding(2U * 4096U, 0.125F);
    std::vector<std::int32_t> mask = {1, 1};
    std::vector<float> noise(32U * 7U, 0.5F);
    const auto image = [](const char * name,
                          const std::vector<std::uint8_t> & pixels) {
        return wam::ImageView{name, wam::ImageEncoding::rgb_u8, pixels.data(),
                              pixels.size(), 224, 224, 3, 224U * 3U};
    };
    const auto f32 = [](const std::vector<float> & values,
                        std::vector<std::int64_t> shape,
                        const char * layout) {
        return wam::TensorView{
            values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), layout, wam::ByteOrder::little};
    };
    wam::Inputs inputs;
    inputs.images = {image("wrist", wrist), image("scene", scene)};
    inputs.state = f32(state, {8}, "D");
    inputs.action_noise = f32(noise, {32, 7}, "T,A");
    inputs.language = wam::EmbeddingInput{
        f32(embedding, {2, 4096}, "T,D"),
        {mask.data(), mask.size() * sizeof(std::int32_t), wam::DType::i32,
         {2}, "T", wam::ByteOrder::little}};
    std::mt19937 rng(31);
    const auto prepared = wam::internal::fastwam::prepare_inputs(
        inputs, *artifact, *policy_spec,
        wam::LanguageRuntimeMode::external_embedding, rng);
    require(prepared.composite_image.width == 448 &&
                prepared.composite_image.height == 224 &&
                prepared.model_state.size() == 8 &&
                prepared.embedding.shape ==
                    std::vector<std::int64_t>({2, 4096}) &&
                prepared.action_noise == noise,
            "FastWAM common policy input boundary changed");
    wam::test::require_error(
        [&] {
            wam::Inputs missing = inputs;
            missing.images.pop_back();
            (void) wam::internal::fastwam::prepare_inputs(
                missing, *artifact, *policy_spec,
                wam::LanguageRuntimeMode::external_embedding, rng);
        },
        wam::ErrorCode::invalid_argument, "missing FastWAM named view");

    wam::ModelOptions options;
    options.artifact_path = argv[1];
    options.backend = wam::Backend::cpu_metadata;
    wam::Model * model = wam::model_load(options);
    const wam::ModelInfo & info = wam::model_info(model);
    require(info.architecture == "fastwam" &&
                info.artifact_policy ==
                    "fastwam_libero_2cam224_minmax" &&
                info.backend == wam::Backend::cpu_metadata &&
                info.language_mode ==
                    wam::LanguageRuntimeMode::external_embedding,
            "FastWAM public metadata lifecycle changed");
    require(info.capabilities.raw_images &&
                info.capabilities.precomputed_embedding &&
                info.capabilities.explicit_action_noise &&
                !info.capabilities.action,
            "FastWAM metadata capabilities changed");
    wam::model_free(model);
    return 0;
}
