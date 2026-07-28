#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace wam::internal::artifact {

inline constexpr std::uint32_t kBundleManifestSchemaVersion = 1;

struct ArtifactBundle {
    std::filesystem::path input_path;
    std::filesystem::path root;
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> tokenizer_path;
    std::optional<std::filesystem::path> language_encoder_path;
    bool uses_manifest = false;
};

ArtifactBundle resolve_bundle(const std::string & input_path);

} // namespace wam::internal::artifact
