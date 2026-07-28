#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace wam {

enum class Backend : std::uint32_t {
    unknown = 0,
    automatic,
    cuda,
    cpu_metadata,
    cpu,
};

enum class ComputePrecision : std::uint32_t {
    unknown = 0,
    automatic,
    f32,
    f16,
    bf16,
    fp8_e4m3,
    fp8_e5m2,
    int8,
};

enum class LanguageRuntimeMode : std::uint32_t {
    automatic = 0,
    tokens,
    external_embedding,
};

enum class LogLevel : std::uint32_t {
    error = 0,
    warning,
    info,
    debug,
};

using LoggerCallback =
    std::function<void(LogLevel level, std::string_view message)>;

struct FixedPrompt {
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> attention_mask;
};

struct DebugDumpConfig {
    bool enabled = false;
    std::string directory;
};

struct RuntimeConfig {
    Backend backend = Backend::automatic;
    ComputePrecision compute_precision = ComputePrecision::automatic;
    std::int32_t device_index = 0;
    std::size_t cpu_thread_count = 0;
    std::size_t prompt_cache_capacity = 0;
    LanguageRuntimeMode language_mode = LanguageRuntimeMode::automatic;
    std::optional<FixedPrompt> fixed_prompt;
    LogLevel log_level = LogLevel::warning;
    LoggerCallback logger;
    DebugDumpConfig debug_dump;
};

struct SessionConfig {
    bool enable_prefix_cache = true;
    std::uint64_t random_seed = 0;
};

} // namespace wam
