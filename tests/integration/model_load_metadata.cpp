#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "wam/wam.h"

int main() {
    wam::test::TempFile valid_file("model-load-valid");
    wam::test::MetadataFixture valid = wam::test::valid_policy_fixture();
    valid.set_string("general.architecture", "fastwam");
    valid.write(valid_file.string());

    wam::RuntimeConfig options;
    wam::test::require_error(
        [&] { (void) wam::Model::load(valid_file.string(), options); },
        wam::ErrorCode::incompatible_artifact,
        "valid PolicySpec must reach FastWAM artifact validation");

    wam::test::TempFile malformed_file("model-load-malformed");
    wam::test::MetadataFixture malformed = wam::test::valid_policy_fixture();
    malformed.set_string("general.architecture", "fastwam");
    malformed.set_string("wam.output.action.recovery.kind", "guess");
    malformed.write(malformed_file.string());
    wam::RuntimeConfig malformed_options;
    wam::test::require_error(
        [&] {
            (void) wam::Model::load(malformed_file.string(), malformed_options);
        },
        wam::ErrorCode::incompatible_artifact,
        "malformed PolicySpec must fail before the architecture factory");

    wam::test::TempFile unknown_file("model-load-unknown-arch");
    wam::test::MetadataFixture unknown = wam::test::valid_policy_fixture();
    unknown.set_string("general.architecture", "pi05");
    unknown.write(unknown_file.string());
    wam::RuntimeConfig unknown_options;
    wam::test::require_error(
        [&] { (void) wam::Model::load(unknown_file.string(), unknown_options); },
        wam::ErrorCode::unsupported, "unknown architecture must fail fast");
    return 0;
}
