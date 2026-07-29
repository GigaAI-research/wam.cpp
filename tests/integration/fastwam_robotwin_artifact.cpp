#include "artifact/artifact_view.h"
#include "models/fastwam/contract.h"
#include "models/fastwam/inputs.h"
#include "policy/policy_spec.h"
#include "support/test_utils.h"
#include "wam/wam.h"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using wam::internal::GgufReader;
using wam::internal::policy::NormalizationKind;
using wam::test::require;

int main(int argc, char ** argv) {
    require(argc == 2, "FastWAM RoboTwin artifact test requires one GGUF path");
    auto artifact_view = wam::internal::artifact::ArtifactView::open(argv[1]);
    auto reader = artifact_view.shared_gguf();
    const auto policy_spec =
        wam::internal::policy::try_read_policy_spec(artifact_view);
    require(policy_spec.has_value(), "FastWAM RoboTwin PolicySpec is missing");
    const auto artifact = wam::internal::fastwam::load_contract(
        reader, *policy_spec);

    require(policy_spec->identity.profile ==
                "fastwam_robotwin_3cam384_zscore",
            "FastWAM RoboTwin profile identity changed");
    require(artifact->geometry.image_height == 384 &&
                artifact->geometry.image_width == 320 &&
                artifact->geometry.num_cameras == 3,
            "FastWAM RoboTwin image geometry changed");
    require(artifact->geometry.proprio_dim == 14 &&
                artifact->geometry.action_dim == 14 &&
                artifact->geometry.action_horizon == 32,
            "FastWAM RoboTwin policy geometry changed");
    require(policy_spec->state.normalization.kind == NormalizationKind::z_score &&
                policy_spec->action.normalization.kind ==
                    NormalizationKind::z_score,
            "FastWAM RoboTwin normalization changed");
    require(policy_spec->state.normalization.output_clamp_lower == -5.0F &&
                policy_spec->state.normalization.output_clamp_upper == 5.0F &&
                !policy_spec->action.normalization.output_clamp_lower.has_value() &&
                !policy_spec->action.normalization.output_clamp_upper.has_value(),
            "FastWAM RoboTwin normalization clamp changed");
    require(policy_spec->action.recovery.kind ==
                wam::internal::policy::ActionRecoveryKind::identity,
            "FastWAM RoboTwin action recovery changed");

    std::vector<std::uint8_t> image_data(480U * 640U * 3U, 127);
    std::vector<float> state(14U, 0.0F);
    std::vector<float> embedding(2U * 4096U, 0.125F);
    std::vector<std::int32_t> mask = {1, 1};
    std::vector<float> noise(32U * 14U, 0.5F);
    const auto image = [&](const char * name) {
        return wam::ImageView{name, wam::ImageEncoding::rgb_u8,
                              image_data.data(), image_data.size(),
                              640, 480, 3, 640U * 3U};
    };
    const auto f32 = [](const std::vector<float> & values,
                        std::vector<std::int64_t> shape,
                        const char * layout) {
        return wam::TensorView{
            values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), layout, wam::ByteOrder::little};
    };
    wam::Observation inputs;
    inputs.images = {image("camera_right_wrist"), image("camera_high"),
                     image("camera_left_wrist")};
    inputs.state = f32(state, {14}, "D");
    inputs.action_noise = f32(noise, {32, 14}, "T,A");
    inputs.language = wam::EmbeddingInput{
        f32(embedding, {2, 4096}, "T,D"),
        {mask.data(), mask.size() * sizeof(std::int32_t), wam::DType::i32,
         {2}, "T", wam::ByteOrder::little}};
    std::mt19937 rng(31);
    const auto prepared = wam::internal::fastwam::prepare_inputs(
        inputs, *artifact, *policy_spec,
        wam::LanguageRuntimeMode::external_embedding, rng);
    require(prepared.observation.composite_image.width == 320 &&
                prepared.observation.composite_image.height == 384 &&
                prepared.observation.model_state.size() == 14 &&
                prepared.observation.action_noise == noise,
            "FastWAM RoboTwin policy input boundary changed");

    wam::RuntimeConfig options;
    options.backend = wam::Backend::cpu_metadata;
    wam::Model model = wam::Model::load(argv[1], options);
    const wam::ModelInfo & info = model.info();
    require(info.architecture == "fastwam" &&
                info.policy_spec != nullptr &&
                info.policy_spec->identity.profile ==
                    "fastwam_robotwin_3cam384_zscore" &&
                info.language_mode ==
                    wam::LanguageRuntimeMode::external_embedding,
            "FastWAM RoboTwin public metadata lifecycle changed");
    return 0;
}
