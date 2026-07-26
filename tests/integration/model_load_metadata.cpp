#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "wam/wam.h"

int main() {
    wam::test::TempFile valid_file("model-load-valid");
    wam::test::MetadataFixture valid = wam::test::valid_policy_fixture();
    valid.set_string("general.architecture", "fastwam");
    valid.write(valid_file.string());

    wam::ModelOptions options;
    options.artifact_path = valid_file.string();
    wam::test::require_error(
        [&] { (void) wam::model_load(options); },
        wam::ErrorCode::incompatible_artifact,
        "valid PolicySpec must reach FastWAM artifact validation");

    wam::test::TempFile malformed_file("model-load-malformed");
    wam::test::MetadataFixture malformed = wam::test::valid_policy_fixture();
    malformed.set_string("general.architecture", "fastwam");
    malformed.set_string("wam.output.action.recovery.kind", "guess");
    malformed.write(malformed_file.string());
    wam::ModelOptions malformed_options;
    malformed_options.artifact_path = malformed_file.string();
    wam::test::require_error(
        [&] { (void) wam::model_load(malformed_options); },
        wam::ErrorCode::incompatible_artifact,
        "malformed PolicySpec must fail before the architecture factory");

    wam::test::TempFile unknown_file("model-load-unknown-arch");
    wam::test::MetadataFixture unknown = wam::test::valid_policy_fixture();
    unknown.set_string("general.architecture", "pi05");
    unknown.write(unknown_file.string());
    wam::ModelOptions unknown_options;
    unknown_options.artifact_path = unknown_file.string();
    wam::test::require_error(
        [&] { (void) wam::model_load(unknown_options); },
        wam::ErrorCode::unsupported, "unknown architecture must fail fast");
    return 0;
}
