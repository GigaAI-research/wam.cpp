#include "arch.h"

namespace wam::internal {

std::string_view arch_name(Arch arch) noexcept {
    switch (arch) {
        case Arch::gwp05:
            return "gwp05";
        case Arch::fastwam:
            return "fastwam";
        case Arch::unknown:
            return "unknown";
    }
    return "unknown";
}

Arch arch_from_name(std::string_view name) noexcept {
    if (name == "gwp05") {
        return Arch::gwp05;
    }
    if (name == "fastwam") {
        return Arch::fastwam;
    }
    return Arch::unknown;
}

} // namespace wam::internal
