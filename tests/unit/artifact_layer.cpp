#include "fixtures/metadata_fixture.h"
#include "support/test_utils.h"

#include "artifact/artifact_view.h"
#include "artifact/manifest.h"
#include "artifact/tensor_spec.h"
#include "policy/policy_spec.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

namespace {

class TempDirectory final {
public:
    explicit TempDirectory(const std::string & stem) {
        static std::atomic<unsigned long> sequence{0};
        path_ = std::filesystem::temp_directory_path() /
            ("wam-" + stem + "-" + std::to_string(getpid()) + "-" +
             std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path & path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    if (!output) throw std::runtime_error("cannot write test file");
}

void test_direct_and_bundle() {
    using wam::test::require;

    TempDirectory directory("artifact-layer");
    const auto model_path = directory.path() / "model.gguf";
    wam::test::MetadataFixture fixture = wam::test::valid_policy_fixture();
    fixture.write(model_path.string());

    const auto direct = wam::internal::artifact::ArtifactView::open(
        model_path.string());
    require(!direct.bundle().uses_manifest,
            "direct GGUF was incorrectly treated as a bundle");

    write_text(directory.path() / "tokenizer.json", "{}\n");
    write_text(directory.path() / "encoder.gguf", "fixture\n");
    write_text(
        directory.path() / "manifest.json",
        "{\n"
        "  \"format\": \"wam-bundle-v1\",\n"
        "  \"manifest_schema_version\": 1,\n"
        "  \"model\": \"model.gguf\",\n"
        "  \"tokenizer\": \"tokenizer.json\",\n"
        "  \"language_encoder\": \"encoder.gguf\"\n"
        "}\n");
    const auto bundle = wam::internal::artifact::ArtifactView::open(
        directory.path().string());
    require(bundle.bundle().uses_manifest &&
                bundle.bundle().tokenizer_path.has_value() &&
                bundle.bundle().language_encoder_path.has_value(),
            "bundle resources were not resolved");
    const auto spec = wam::internal::policy::try_read_policy_spec(bundle);
    require(spec.has_value() && spec->identity.artifact_schema_version == 3,
            "bundle PolicySpec v3 was not loaded");

    using wam::internal::artifact::TensorSpec;
    wam::internal::artifact::validate_tensor(
        bundle, TensorSpec{"wam.norm.action.upper", {wam::DType::f32},
                           std::vector<std::int64_t>{4}, true});
    wam::internal::artifact::validate_tensor(
        bundle, TensorSpec{"optional.tensor", {wam::DType::bf16},
                           std::nullopt, false});
    wam::test::require_error(
        [&] {
            wam::internal::artifact::validate_tensor(
                bundle, TensorSpec{"wam.norm.action.upper",
                                   {wam::DType::bf16}, std::nullopt, true});
        },
        wam::ErrorCode::incompatible_artifact, "TensorSpec dtype mismatch");
    wam::test::require_error(
        [&] {
            wam::internal::artifact::validate_tensor(
                bundle, TensorSpec{"wam.norm.action.upper",
                                   {wam::DType::f32},
                                   std::vector<std::int64_t>{3}, true});
        },
        wam::ErrorCode::incompatible_artifact, "TensorSpec shape mismatch");
}

void test_manifest_failures() {
    using wam::test::require_error;

    TempDirectory missing_manifest("bundle-missing-manifest");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                missing_manifest.path().string());
        },
        wam::ErrorCode::not_found, "missing bundle manifest");

    TempDirectory invalid("bundle-invalid");
    write_text(invalid.path() / "manifest.json", "{not-json}\n");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::incompatible_artifact, "malformed bundle manifest");

    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":2,\"model\":\"model.gguf\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::incompatible_artifact, "bundle schema mismatch");

    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":1,\"model\":\"missing.gguf\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::not_found, "missing bundle model");

    write_text(invalid.path() / "model.gguf", "fixture\n");
    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":1,\"model\":\"model.gguf\","
               "\"tokenizer\":\"missing.json\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::not_found, "missing optional bundle resource");

    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":1,\"model\":\"../escape.gguf\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::incompatible_artifact, "bundle parent traversal");

    TempDirectory outside("bundle-outside");
    write_text(outside.path() / "outside.gguf", "fixture\n");
    std::filesystem::create_symlink(outside.path() / "outside.gguf",
                                    invalid.path() / "linked.gguf");
    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":1,\"model\":\"linked.gguf\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::incompatible_artifact, "bundle symlink escape");

    write_text(invalid.path() / "manifest.json",
               "{\"format\":\"wam-bundle-v1\","
               "\"manifest_schema_version\":1,\"model\":\"model.gguf\","
               "\"extra\":\"unsupported\"}");
    require_error(
        [&] {
            (void) wam::internal::artifact::resolve_bundle(
                invalid.path().string());
        },
        wam::ErrorCode::incompatible_artifact, "unknown manifest field");
}

void test_policy_schema_compatibility() {
    using wam::test::require;

    TempDirectory directory("policy-schema");
    const auto legacy_path = directory.path() / "legacy-v2.gguf";
    auto legacy = wam::test::valid_policy_fixture();
    legacy.set_u32("wam.artifact_schema_version", 2);
    legacy.remove("wam.input.image.resample_boundary");
    legacy.write(legacy_path.string());
    const auto legacy_view =
        wam::internal::artifact::ArtifactView::open(legacy_path.string());
    const auto legacy_spec =
        wam::internal::policy::try_read_policy_spec(legacy_view);
    require(legacy_spec.has_value() &&
                legacy_spec->images.resample_boundary ==
                    wam::ResampleBoundaryMode::truncate,
            "PolicySpec v2 compatibility default changed");

    const auto invalid_path = directory.path() / "invalid-v3.gguf";
    auto invalid = wam::test::valid_policy_fixture();
    invalid.remove("wam.input.image.resample_boundary");
    invalid.write(invalid_path.string());
    const auto invalid_view =
        wam::internal::artifact::ArtifactView::open(invalid_path.string());
    wam::test::require_error(
        [&] {
            (void) wam::internal::policy::try_read_policy_spec(invalid_view);
        },
        wam::ErrorCode::incompatible_artifact,
        "PolicySpec v3 missing resample boundary");
}

} // namespace

int main() {
    test_direct_and_bundle();
    test_manifest_failures();
    test_policy_schema_compatibility();
    return 0;
}
