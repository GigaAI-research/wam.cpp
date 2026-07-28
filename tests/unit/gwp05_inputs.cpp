#include "fixtures/metadata_fixture.h"
#include "artifact/artifact_view.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "artifact/gguf_reader.h"
#include "models/gwp05/artifact.h"
#include "models/gwp05/inputs.h"
#include "policy/policy_spec.h"

#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         std::string layout = {}) {
    wam::TensorView view;
    view.data = values.data();
    view.byte_size = values.size() * sizeof(float);
    view.dtype = wam::DType::f32;
    view.shape = std::move(shape);
    view.layout = std::move(layout);
    view.byte_order = wam::ByteOrder::little;
    return view;
}

wam::TensorView i32_view(const std::vector<std::int32_t> & values,
                         std::vector<std::int64_t> shape) {
    wam::TensorView view;
    view.data = values.data();
    view.byte_size = values.size() * sizeof(std::int32_t);
    view.dtype = wam::DType::i32;
    view.shape = std::move(shape);
    view.byte_order = wam::ByteOrder::little;
    return view;
}

wam::ImageView image(const std::string & name,
                     const std::array<std::uint8_t, 12> & pixels) {
    wam::ImageView view;
    view.name = name;
    view.encoding = wam::ImageEncoding::rgb_u8;
    view.data = pixels.data();
    view.byte_size = pixels.size();
    view.width = 2;
    view.height = 2;
    view.channels = 3;
    return view;
}

} // namespace

int main() {
    using wam::test::require;
    using wam::test::require_error;

    wam::test::TempFile file("gwp05-inputs");
    wam::test::MetadataFixture fixture =
        wam::test::valid_gwp05_policy_fixture();
    fixture.write(file.string());
    const auto artifact_view =
        wam::internal::artifact::ArtifactView::open(file.string());
    const auto reader = artifact_view.shared_gguf();
    const auto spec =
        *wam::internal::policy::try_read_policy_spec(artifact_view);
    const auto artifact = wam::internal::gwp05::load_artifact(reader, spec);

    const std::array<std::uint8_t, 12> scene{};
    const std::array<std::uint8_t, 12> left{};
    const std::array<std::uint8_t, 12> right{};
    std::vector<float> state(14, 0.25F);
    std::vector<float> noise(48 * 32, 0.5F);
    std::vector<std::int32_t> tokens = {5, 6, 0};
    std::vector<std::int32_t> token_mask = {1, 1, 0};

    wam::Observation inputs;
    inputs.images = {image("right_wrist", right), image("scene", scene),
                     image("left_wrist", left)};
    inputs.state = f32_view(state, {14}, "D");
    inputs.language = wam::TokenInput{
        wam::ArrayView<std::int32_t>(tokens),
        wam::ArrayView<std::int32_t>(token_mask)};
    inputs.action_noise = f32_view(noise, {48, 32}, "T,A");
    std::mt19937 session_rng(31);

    const wam::internal::gwp05::PreparedInputs prepared =
        wam::internal::gwp05::prepare_inputs(
            inputs, *artifact, spec, wam::LanguageRuntimeMode::tokens,
            session_rng);
    require(prepared.observation.composite_image.width ==
                spec.images.composition.width &&
                prepared.observation.composite_image.height ==
                    spec.images.composition.height &&
                prepared.observation.composite_image.layout ==
                    wam::internal::policy::TensorLayout::chw,
            "GWP images were not prepared by common image ops");
    require(prepared.observation.raw_state == state,
            "GWP raw state copy changed");
    require(prepared.observation.model_state.size() == spec.state.model_dim,
            "GWP model state was not padded and normalized");
    require(prepared.token_ids == std::vector<std::int32_t>({5, 6}) &&
                prepared.attention_mask ==
                    std::vector<std::int32_t>({1, 1}),
            "GWP token padding was not canonicalized");
    require(prepared.observation.action_noise == noise,
            "GWP explicit action noise was not retained");

    std::vector<float> embedding_values(2 * 64, 0.125F);
    std::vector<std::int32_t> embedding_mask = {1, 1};
    wam::Observation embedding_inputs = inputs;
    embedding_inputs.action_noise = {};
    wam::EmbeddingInput embedding;
    embedding.embedding =
        f32_view(embedding_values, {2, 64}, "T,D");
    embedding.attention_mask = i32_view(embedding_mask, {2});
    embedding_inputs.language = embedding;
    const auto prepared_embedding =
        wam::internal::gwp05::prepare_inputs(
            embedding_inputs, *artifact, spec,
            wam::LanguageRuntimeMode::external_embedding,
            session_rng);
    require(prepared_embedding.embedding.shape ==
                std::vector<std::int64_t>({2, 64}) &&
                prepared_embedding.observation.action_noise.size() == 48U * 32U,
            "GWP external embedding contract changed");

    {
        wam::Observation bad = inputs;
        bad.images.pop_back();
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument, "missing GWP image");
    }
    {
        wam::Observation bad = inputs;
        bad.images[0].name = "scene";
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument, "duplicate GWP image");
    }
    {
        wam::Observation bad = inputs;
        bad.images[0].encoding = wam::ImageEncoding::jpeg;
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::unsupported, "encoded GWP image");
    }
    {
        wam::Observation bad = inputs;
        bad.state.dtype = wam::DType::bf16;
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument, "wrong GWP state dtype");
    }
    {
        std::vector<float> nonfinite_state = state;
        nonfinite_state[4] = std::numeric_limits<float>::infinity();
        wam::Observation bad = inputs;
        bad.state = f32_view(nonfinite_state, {14}, "D");
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument, "non-finite GWP state");
    }
    {
        std::vector<std::int32_t> bad_mask = {1, 0, 1};
        wam::Observation bad = inputs;
        bad.language = wam::TokenInput{
            wam::ArrayView<std::int32_t>(tokens),
            wam::ArrayView<std::int32_t>(bad_mask)};
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument,
            "noncontiguous GWP token mask");
    }
    {
        std::vector<std::int32_t> bad_tokens = {5, 128};
        std::vector<std::int32_t> bad_mask = {1, 1};
        wam::Observation bad = inputs;
        bad.language = wam::TokenInput{
            wam::ArrayView<std::int32_t>(bad_tokens),
            wam::ArrayView<std::int32_t>(bad_mask)};
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument,
            "out-of-vocabulary GWP token");
    }
    {
        wam::Observation bad = inputs;
        bad.action_noise.shape = {1, 48, 32};
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument,
            "batched GWP action noise");
    }
    {
        std::vector<float> nonfinite_noise = noise;
        nonfinite_noise[0] = std::numeric_limits<float>::quiet_NaN();
        wam::Observation bad = inputs;
        bad.action_noise = f32_view(nonfinite_noise, {48, 32}, "T,A");
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::invalid_argument,
            "non-finite GWP action noise");
    }
    {
        wam::Observation bad = inputs;
        bad.history.push_back({"reference_latent", {}});
        require_error(
            [&] {
                (void) wam::internal::gwp05::prepare_inputs(
                    bad, *artifact, spec, wam::LanguageRuntimeMode::tokens,
                    session_rng);
            },
            wam::ErrorCode::unsupported, "GWP history input");
    }
    require_error(
        [&] {
            (void) wam::internal::gwp05::prepare_inputs(
                inputs, *artifact, spec,
                wam::LanguageRuntimeMode::external_embedding,
                session_rng);
        },
        wam::ErrorCode::invalid_argument,
        "token payload in embedding mode");
    return 0;
}
