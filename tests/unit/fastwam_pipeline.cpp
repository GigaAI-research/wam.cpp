#include "models/fastwam/contract.h"
#include "models/fastwam/inputs.h"
#include "models/fastwam/networks/proprio_projector.h"
#include "models/fastwam/pipeline.h"
#include "models/fastwam/state.h"
#include "support/test_utils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

wam::internal::policy::PolicySpec make_policy_spec() {
    wam::internal::policy::PolicySpec spec;
    for (const char * role : {"scene", "wrist"}) {
        wam::internal::policy::ImageTransformSpec view;
        view.role = role;
        view.target_height = 2;
        view.target_width = 2;
        view.resize = wam::internal::policy::ResizeMode::none;
        spec.images.views.push_back(std::move(view));
    }
    spec.images.composition.kind = wam::internal::policy::ImageCompositionKind::canvas;
    spec.images.composition.height = 2;
    spec.images.composition.width = 4;
    spec.images.composition.placements = {
        {"scene", 0, 0, 2, 2}, {"wrist", 2, 0, 2, 2}};
    spec.images.pixel_range = wam::internal::policy::PixelRange::minus_one_to_one;
    spec.images.tensor_layout = wam::internal::policy::TensorLayout::chw;
    spec.state.real_dim = 2;
    spec.state.model_dim = 2;
    spec.language.input_mode = wam::internal::policy::LanguageInputMode::embedding;
    spec.language.max_tokens = 2;
    spec.language.attention_mask_required = true;
    spec.action.horizon = 2;
    spec.action.real_dim = 2;
    spec.action.model_dim = 2;
    return spec;
}

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         const char * layout) {
    return {values.data(), values.size() * sizeof(float), wam::DType::f32,
            std::move(shape), layout, wam::ByteOrder::little};
}

} // namespace

int main() {
    using namespace wam::internal::fastwam;
    using wam::test::require;
    using wam::test::require_error;

    FastWamContract contract;
    contract.geometry.image_height = 2;
    contract.geometry.image_width = 4;
    contract.geometry.num_cameras = 2;
    contract.geometry.proprio_dim = 2;
    contract.geometry.action_dim = 2;
    contract.geometry.action_horizon = 2;
    contract.geometry.context_len = 2;
    contract.geometry.text_dim = 2;
    contract.proprio_weight = {1.0F, 2.0F, 3.0F, 4.0F};
    contract.proprio_bias = {0.5F, -0.5F};

    const std::vector<ggml_bf16_t> projected =
        project_proprio({2.0F, 3.0F}, contract);
    require(projected.size() == 2 &&
                std::fabs(ggml_bf16_to_fp32(projected[0]) - 8.5F) < 0.01F &&
                std::fabs(ggml_bf16_to_fp32(projected[1]) - 17.5F) < 0.01F,
            "FastWAM proprio projection changed");

    ModelOptions options;
    options.backend = wam::Backend::cpu_metadata;
    require_error(
        [&] { (void) load_model_resources(contract, options); },
        wam::ErrorCode::unsupported,
        "FastWAM compute resources must reject metadata-only backend");
    options.backend = wam::Backend::automatic;
    options.compute_precision = wam::ComputePrecision::f32;
    require_error(
        [&] { (void) load_model_resources(contract, options); },
        wam::ErrorCode::unsupported,
        "FastWAM compute resources must reject F32 precision");

    auto resources = std::make_shared<ModelResources>(
        std::make_shared<wam::internal::runtime::Logger>(wam::RuntimeConfig{}),
        std::make_shared<wam::internal::ggml_backend::DebugDump>());
    auto first_state = create_session_state();
    auto second_state = create_session_state();
    require(first_state.get() != second_state.get(),
            "FastWAM Sessions must own distinct mutable state");
    first_state->prediction_count = 3;
    reset_session(*resources, *first_state);
    require(first_state->prediction_count == 0 &&
                second_state->prediction_count == 0,
            "FastWAM reset or Session state isolation changed");

    const wam::internal::policy::PolicySpec spec = make_policy_spec();
    const std::array<std::uint8_t, 12> scene{};
    const std::array<std::uint8_t, 12> wrist{};
    const auto image = [](const char * name,
                          const std::array<std::uint8_t, 12> & pixels) {
        return wam::ImageView{name, wam::ImageEncoding::rgb_u8, pixels.data(),
                              pixels.size(), 2, 2, 3, 6};
    };
    const std::vector<float> state = {0.25F, -0.25F};
    const std::vector<float> embedding = {0.1F, 0.2F};
    const std::vector<std::int32_t> mask = {1};
    const std::vector<float> noise = {0.5F, 0.25F, -0.25F, -0.5F};
    wam::Observation input;
    input.images = {image("wrist", wrist), image("scene", scene)};
    input.state = f32_view(state, {2}, "D");
    input.language = wam::EmbeddingInput{
        f32_view(embedding, {1, 2}, "T,D"),
        {mask.data(), mask.size() * sizeof(std::int32_t), wam::DType::i32,
         {1}, "T", wam::ByteOrder::little}};
    input.action_noise = f32_view(noise, {2, 2}, "T,A");

    std::mt19937 explicit_rng(77);
    std::mt19937 explicit_baseline(77);
    const PreparedInputs explicit_prepared = prepare_inputs(
        input, contract, spec, wam::LanguageRuntimeMode::external_embedding,
        explicit_rng);
    require(explicit_prepared.observation.action_noise == noise &&
                explicit_rng() == explicit_baseline(),
            "explicit FastWAM noise must be preserved without consuming RNG");

    input.action_noise = {};
    std::mt19937 first_rng(91);
    std::mt19937 second_rng(91);
    const PreparedInputs first_generated = prepare_inputs(
        input, contract, spec, wam::LanguageRuntimeMode::external_embedding,
        first_rng);
    const PreparedInputs second_generated = prepare_inputs(
        input, contract, spec, wam::LanguageRuntimeMode::external_embedding,
        second_rng);
    require(first_generated.observation.action_noise ==
                second_generated.observation.action_noise,
            "equal FastWAM Session seeds must reproduce generated noise");
    return 0;
}
