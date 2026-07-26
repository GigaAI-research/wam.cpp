#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "models/common/gguf_reader.h"
#include "policy/policy_spec.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

void expect_failure(wam::test::MetadataFixture fixture,
                    const std::string & stem,
                    const std::string & expected_field) {
    wam::test::TempFile file(stem);
    fixture.write(file.string());
    const auto reader = wam::internal::GgufReader::open(file.string());
    try {
        (void) wam::internal::policy::try_read_policy_spec_draft(*reader);
    } catch (const wam::Error & error) {
        wam::test::require(
            error.code() == wam::ErrorCode::incompatible_artifact,
            stem + ": wrong error code");
        wam::test::require(!error.details().empty(),
                           stem + ": missing structured error detail");
        wam::test::require(error.details().front().field == expected_field,
                           stem + ": wrong error field: " +
                               error.details().front().field);
        return;
    }
    throw std::runtime_error(stem + ": expected PolicySpec failure");
}

} // namespace

int main() {
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.remove("wam.artifact_schema_version");
        expect_failure(std::move(fixture), "missing-schema",
                       "wam.artifact_schema_version");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.remove("wam.policy.profile");
        expect_failure(std::move(fixture), "missing-profile",
                       "wam.policy.profile");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_u32("wam.policy.profile", 7);
        expect_failure(std::move(fixture), "wrong-metadata-type",
                       "wam.policy.profile");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_u32("wam.artifact_schema_version", 99);
        expect_failure(std::move(fixture), "unknown-schema",
                       "wam.artifact_schema_version");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_string("wam.input.image.scene.resize_mode", "letterbox");
        expect_failure(std::move(fixture), "unknown-resize",
                       "wam.input.image.scene.resize_mode");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_string_array("wam.input.image.roles", {"scene", "scene"});
        fixture.set_u32_array("wam.input.image.composition.scene.rect",
                              {0, 0, 2, 2});
        expect_failure(std::move(fixture), "duplicate-role",
                       "wam.input.image.roles");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_u32_array("wam.input.image.composition.wrist.rect",
                              {3, 0, 2, 2});
        expect_failure(std::move(fixture), "canvas-out-of-bounds",
                       "wam.input.image.composition.wrist.rect");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_u32_array("wam.input.image.composition.wrist.rect",
                              {1, 0, 2, 2});
        expect_failure(std::move(fixture), "canvas-overlap",
                       "wam.input.image.composition.wrist.rect");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_f32_tensor("wam.norm.state.lower", {-1, -2, -3});
        expect_failure(std::move(fixture), "stats-shape",
                       "wam.norm.state.lower");
    }
    {
        auto fixture = wam::test::valid_gwp05_robotwin_14d_policy_fixture();
        fixture.set_bool("wam.normalization.state.clip", true);
        expect_failure(std::move(fixture), "zscore-clip",
                       "wam.normalization.state.clip");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_i32_array(
            "wam.output.action.recovery.reference_state_indices", {0, 1});
        expect_failure(
            std::move(fixture), "recovery-length",
            "wam.output.action.recovery.reference_state_indices");
    }
    {
        auto fixture = wam::test::valid_policy_fixture();
        fixture.set_i32_array(
            "wam.output.action.recovery.reference_state_indices", {0, 3, -1});
        expect_failure(
            std::move(fixture), "recovery-index",
            "wam.output.action.recovery.reference_state_indices");
    }
    return 0;
}
