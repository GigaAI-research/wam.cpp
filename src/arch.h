#pragma once

#include <string_view>

namespace wam::internal {

enum class Arch {
    unknown = 0,
    gwp05,
    fastwam,
};

std::string_view arch_name(Arch arch) noexcept;
Arch arch_from_name(std::string_view name) noexcept;

} // namespace wam::internal
