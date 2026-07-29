#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "wam/wam.h"

#include <string>
#include <utility>

namespace {

wam::Model load(wam::test::MetadataFixture fixture, const std::string & stem,
                wam::RuntimeConfig options = {}) {
    wam::test::TempFile file(stem);
    fixture.write(file.string());
    if (options.backend == wam::Backend::automatic) {
        options.backend = wam::Backend::cpu_metadata;
    }
    try {
        return wam::Model::load(file.string(), options);
    } catch (const wam::Error & error) {
        std::string message = stem + ": " + error.what();
        for (const wam::ErrorDetail & detail : error.details()) {
            message += " [" + detail.field + ": " + detail.reason + "]";
        }
        throw std::runtime_error(message);
    }
}

template <typename Mutator>
void expect_load_error(const std::string & stem, wam::ErrorCode code,
                       Mutator mutator) {
    wam::test::TempFile file(stem);
    wam::test::MetadataFixture fixture =
        wam::test::valid_gwp05_policy_fixture();
    mutator(fixture);
    fixture.write(file.string());
    wam::RuntimeConfig options;
    options.backend = wam::Backend::cpu_metadata;
    wam::test::require_error(
        [&] { (void) wam::Model::load(file.string(), options); }, code, stem);
}

} // namespace

int main() {
    using wam::test::require;

    wam::Model draft = load(wam::test::valid_gwp05_policy_fixture(),
                            "gwp05-draft");
    const wam::ModelInfo & draft_info = draft.info();
    require(draft_info.architecture == "gwp05", "GWP architecture changed");
    require(draft_info.policy_spec != nullptr &&
                draft_info.policy_spec->identity.profile ==
                    "gwp05_robotwin_dual_arm_fixture",
            "draft PolicySpec profile was not retained");
    require(draft_info.backend == wam::Backend::cpu_metadata,
            "metadata backend was not retained");
    require(draft_info.compute_precision == wam::ComputePrecision::f32,
            "automatic precision did not resolve for metadata loading");
    require(draft_info.language_mode == wam::LanguageRuntimeMode::tokens,
            "automatic language mode did not resolve to tokens");
    require(!draft_info.capabilities.action &&
                draft_info.capabilities.raw_images &&
                draft_info.capabilities.token_input &&
                !draft_info.capabilities.precomputed_embedding &&
                draft_info.capabilities.explicit_action_noise,
            "GWP05 metadata capabilities are inconsistent");
    wam::test::require_error(
        [&] { (void) draft.create_session(); },
        wam::ErrorCode::unsupported,
        "metadata-only GWP model must not create an execution session");
    wam::Model robotwin = load(
        wam::test::valid_gwp05_robotwin_14d_policy_fixture(),
        "gwp05-robotwin-14d");
    require(robotwin.info().policy_spec != nullptr &&
                robotwin.info().policy_spec->identity.profile ==
                    "gwp05_robotwin_dual_arm_14d_zscore",
            "audited RoboTwin PolicySpec was not retained");
    require(!robotwin.info().capabilities.action &&
                robotwin.info().capabilities.explicit_action_noise,
            "audited RoboTwin metadata fixture exposed compute unexpectedly");
    wam::Model legacy = load(wam::test::valid_gwp05_legacy_fixture(),
                             "gwp05-legacy");
    require(legacy.info().policy_spec != nullptr &&
                legacy.info().policy_spec->identity.profile ==
                    "legacy-gwp05-dual-arm-32d-quantile",
            "legacy GWP PolicySpec was not constructed");
    wam::RuntimeConfig embedding_options;
    embedding_options.language_mode =
        wam::LanguageRuntimeMode::external_embedding;
    wam::Model embedding = load(
        wam::test::valid_gwp05_policy_fixture(), "gwp05-embedding",
        embedding_options);
    require(embedding.info().language_mode ==
                wam::LanguageRuntimeMode::external_embedding,
            "explicit embedding mode was not retained");
    require(!embedding.info().capabilities.token_input &&
                embedding.info().capabilities.precomputed_embedding,
            "embedding runtime capabilities are inconsistent");
    wam::RuntimeConfig fixed_options;
    fixed_options.fixed_prompt = wam::FixedPrompt{{5, 6}, {1, 1}};
    wam::Model fixed = load(wam::test::valid_gwp05_policy_fixture(),
                            "gwp05-fixed-prompt", fixed_options);
    require(!fixed.info().capabilities.arbitrary_token_input &&
                fixed.info().capabilities.fixed_token_input,
            "fixed prompt capabilities are inconsistent");
    expect_load_error("gwp05-private-dimension", wam::ErrorCode::incompatible_artifact,
                      [](wam::test::MetadataFixture & fixture) {
                          fixture.set_u32("gwp05.action_dim", 31);
                      });
    expect_load_error("gwp05-missing-geometry", wam::ErrorCode::incompatible_artifact,
                      [](wam::test::MetadataFixture & fixture) {
                          fixture.remove("gwp05.hidden");
                      });
    expect_load_error("gwp05-attention-geometry", wam::ErrorCode::incompatible_artifact,
                      [](wam::test::MetadataFixture & fixture) {
                          fixture.set_u32("gwp05.n_heads", 2);
                      });
    expect_load_error("gwp05-view-mismatch", wam::ErrorCode::incompatible_artifact,
                      [](wam::test::MetadataFixture & fixture) {
                          fixture.set_u32("gwp05.num_views", 2);
                      });
    expect_load_error("gwp05-unknown-conversion", wam::ErrorCode::unsupported,
                      [](wam::test::MetadataFixture & fixture) {
                          fixture.set_string("gwp05.conversion_policy", "guess");
                      });

    wam::test::TempFile legacy_shape_file("gwp05-unknown-legacy-shape");
    wam::test::MetadataFixture legacy_shape =
        wam::test::valid_gwp05_legacy_fixture();
    legacy_shape.set_u32("gwp05.action_chunk", 2);
    legacy_shape.write(legacy_shape_file.string());
    wam::RuntimeConfig legacy_shape_options;
    legacy_shape_options.backend = wam::Backend::cpu_metadata;
    wam::test::require_error(
        [&] {
            (void) wam::Model::load(legacy_shape_file.string(),
                                    legacy_shape_options);
        },
        wam::ErrorCode::unsupported,
        "unknown legacy GWP geometry must not be guessed");

    wam::test::TempFile robotwin_legacy_file(
        "gwp05-robotwin-14d-without-policy-spec");
    wam::test::MetadataFixture robotwin_legacy =
        wam::test::valid_gwp05_legacy_fixture();
    robotwin_legacy.set_u32("gwp05.action_dim", 14);
    robotwin_legacy.write(robotwin_legacy_file.string());
    wam::RuntimeConfig robotwin_legacy_options;
    robotwin_legacy_options.backend = wam::Backend::cpu_metadata;
    wam::test::require_error(
        [&] {
            (void) wam::Model::load(robotwin_legacy_file.string(),
                                    robotwin_legacy_options);
        },
        wam::ErrorCode::unsupported,
        "14D RoboTwin artifacts must carry an explicit PolicySpec");

    wam::test::TempFile backend_file("gwp05-metadata-precision");
    wam::test::MetadataFixture backend_fixture =
        wam::test::valid_gwp05_policy_fixture();
    backend_fixture.write(backend_file.string());
    wam::RuntimeConfig backend_options;
    backend_options.backend = wam::Backend::cpu_metadata;
    backend_options.compute_precision = wam::ComputePrecision::bf16;
    wam::test::require_error(
        [&] { (void) wam::Model::load(backend_file.string(), backend_options); },
        wam::ErrorCode::unsupported,
        "metadata-only GWP loading must reject BF16 execution precision");
    return 0;
}
