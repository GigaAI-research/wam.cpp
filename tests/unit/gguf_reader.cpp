#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "models/common/gguf_reader.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

int main() {
    using wam::test::require;
    using wam::test::require_error;

    wam::test::TempFile file("gguf-reader");
    wam::test::MetadataFixture fixture = wam::test::valid_policy_fixture();
    fixture.write(file.string());

    const std::shared_ptr<wam::internal::GgufReader> reader =
        wam::internal::GgufReader::open(file.string());
    require(reader->path() == file.string(), "GGUF path changed");
    require(reader->file_size() > reader->data_offset(),
            "GGUF payload offset is outside the file");
    require(reader->require_string("general.architecture") == "gwp05",
            "string metadata read failed");
    require(reader->require_u32("wam.output.action.horizon") == 2,
            "u32 metadata read failed");
    require(reader->require_bool("wam.input.image.scene.antialias"),
            "bool metadata read failed");
    require(reader->require_string_array("wam.input.image.roles").size() == 2,
            "string array metadata read failed");
    require(reader->require_i32_array(
                "wam.output.action.recovery.reference_state_indices")[2] == -1,
            "i32 array metadata read failed");
    require(reader->require_u32_array(
                "wam.input.image.composition.wrist.rect")[0] == 2,
            "u32 array metadata read failed");
    require(reader->tensor_count() == 4, "fixture tensor count changed");
    require(reader->require_tensor("wam.norm.action.q99").dtype ==
                wam::DType::f32,
            "tensor dtype mapping failed");
    require(reader->read_f32_tensor("wam.norm.action.q99")[2] == 3.0F,
            "F32 tensor payload read failed");
    reader->require_shape("wam.norm.action.q99", {4});

    require_error(
        [&] { (void) reader->require_string("missing.key"); },
        wam::ErrorCode::incompatible_artifact, "missing metadata key");
    require_error(
        [&] { (void) reader->require_u32("general.architecture"); },
        wam::ErrorCode::incompatible_artifact, "wrong metadata type");
    require_error(
        [&] { reader->require_shape("wam.norm.action.q99", {3}); },
        wam::ErrorCode::incompatible_artifact, "wrong tensor shape");
    require_error(
        [&] { (void) reader->read_f32_tensor("missing.tensor"); },
        wam::ErrorCode::incompatible_artifact, "missing tensor");

    require_error(
        [&] {
            (void) wam::internal::GgufReader::open(
                file.string() + ".does-not-exist");
        },
        wam::ErrorCode::not_found, "missing artifact");

    wam::test::TempFile truncated_file("gguf-truncated");
    wam::test::MetadataFixture truncated =
        wam::test::valid_policy_fixture();
    truncated.write(truncated_file.string());
    std::uint64_t truncated_size = 0;
    {
        const auto descriptor =
            wam::internal::GgufReader::open(truncated_file.string());
        const wam::internal::GgufTensorInfo & last =
            descriptor->tensors().back();
        truncated_size = last.file_offset + last.bytes - 1;
    }
    std::filesystem::resize_file(truncated_file.string(), truncated_size);
    require_error(
        [&] {
            (void) wam::internal::GgufReader::open(truncated_file.string());
        },
        wam::ErrorCode::incompatible_artifact, "truncated tensor payload");
    return 0;
}
