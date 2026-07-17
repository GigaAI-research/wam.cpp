#pragma once

#include <cstdint>

namespace wam {

struct Version {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
};

// These version domains intentionally evolve independently.
inline constexpr Version kRuntimeVersion{0, 1, 0};
inline constexpr Version kProtocolVersion{1, 1, 0};
inline constexpr std::uint32_t kArtifactSchemaVersion = 1;
inline constexpr std::uint32_t kAbiVersion = 2;

inline constexpr const char * kRuntimeVersionString = "0.1.0";
inline constexpr const char * kProtocolVersionString = "1.1.0";

inline constexpr Version runtime_version() noexcept { return kRuntimeVersion; }
inline constexpr Version protocol_version() noexcept { return kProtocolVersion; }
inline constexpr std::uint32_t artifact_schema_version() noexcept {
    return kArtifactSchemaVersion;
}
inline constexpr std::uint32_t abi_version() noexcept { return kAbiVersion; }

} // namespace wam
