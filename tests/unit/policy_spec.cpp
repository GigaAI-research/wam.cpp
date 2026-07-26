#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "models/common/gguf_reader.h"
#include "policy/policy_spec.h"

int main() {
    using wam::test::require;

    wam::test::TempFile file("policy-spec");
    wam::test::MetadataFixture fixture = wam::test::valid_policy_fixture();
    fixture.write(file.string());
    const auto reader = wam::internal::GgufReader::open(file.string());

    const auto loaded =
        wam::internal::policy::try_read_policy_spec_draft(*reader);
    require(loaded.has_value(), "valid PolicySpec was not detected");
    const wam::internal::policy::PolicySpecDraft & spec = *loaded;
    require(spec.identity.artifact_schema_version == 2,
            "schema version changed");
    require(spec.identity.profile == "synthetic_2cam_joint",
            "profile was not loaded");
    require(wam::internal::policy::policy_image_count(spec) == 2,
            "image count was not derived from views");
    require(wam::internal::policy::require_image_spec(spec, "wrist")
                .target_width == 2,
            "named image lookup failed");
    require(spec.images.composition.placements.size() == 2,
            "canvas placements were not loaded");
    require(spec.state.real_dim == 3 && spec.state.model_dim == 4,
            "state dimensions were not loaded");
    require(spec.state.stats.lower.size() == 4,
            "state statistics were not loaded");
    require(spec.language.input_mode ==
                wam::internal::policy::LanguageInputMode::tokens,
            "language input mode was not loaded");
    require(spec.action.horizon == 2 && spec.action.real_dim == 3 &&
                spec.action.model_dim == 4,
            "action geometry was not loaded");
    require(spec.action.recovery.kind ==
                wam::internal::policy::ActionRecoveryKind::add_current_state,
            "action recovery kind was not loaded");
    require(spec.action.recovery.reference_state_indices[2] == -1,
            "action recovery indices were not loaded");

    wam::test::require_error(
        [&] {
            (void) wam::internal::policy::require_image_spec(spec, "missing");
        },
        wam::ErrorCode::invalid_argument, "missing image role lookup");

    wam::test::TempFile legacy_file("policy-absent");
    wam::test::MetadataFixture legacy;
    legacy.set_string("general.architecture", "gwp05");
    legacy.set_f32_tensor("legacy.weight", {1.0F});
    legacy.write(legacy_file.string());
    const auto legacy_reader =
        wam::internal::GgufReader::open(legacy_file.string());
    require(!wam::internal::policy::try_read_policy_spec_draft(*legacy_reader)
                 .has_value(),
            "artifact without schema marker must remain a legacy candidate");
    return 0;
}
