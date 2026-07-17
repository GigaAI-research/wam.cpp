#pragma once

#include <string_view>

namespace wam::internal {

enum class Arch {
    unknown = 0,
    gwp05 = 1,
    fastwam = 2,
};

inline constexpr std::string_view arch_name(Arch arch) noexcept {
    switch (arch) {
        case Arch::gwp05: return "gwp05";
        case Arch::fastwam: return "fastwam";
        case Arch::unknown: return "unknown";
    }
    return "unknown";
}

inline constexpr Arch arch_from_name(std::string_view name) noexcept {
    return name == "gwp05" ? Arch::gwp05
         : name == "fastwam" ? Arch::fastwam
                              : Arch::unknown;
}

} // namespace wam::internal
