#include "support/test_utils.h"

#include "arch.h"
#include "model_registry.h"

#include <memory>
#include <optional>
#include <string>

namespace {

wam::internal::ModelFactory unsupported_factory() {
    return [](const wam::ModelOptions &, wam::ModelInfo,
              std::optional<wam::internal::policy::PolicySpecDraft>,
              std::shared_ptr<wam::internal::GgufReader>)
               -> std::unique_ptr<wam::internal::ModelImpl> {
        throw wam::Error(wam::ErrorCode::unsupported, "test factory");
    };
}

void verify_builtin(wam::internal::Arch arch, bool expected) {
    const wam::internal::ModelFactory * factory =
        wam::internal::model_registry().find(arch);
    wam::test::require((factory != nullptr) == expected,
                       "built-in registry configuration mismatch for " +
                           std::string(wam::internal::arch_name(arch)));
    if (factory != nullptr && arch == wam::internal::Arch::fastwam) {
        wam::test::require_error(
            [&] {
                (*factory)(wam::ModelOptions{}, wam::ModelInfo{}, std::nullopt,
                           nullptr);
            },
            wam::ErrorCode::unsupported,
            "FastWAM factory must remain unsupported");
    }
}

} // namespace

int main() {
    using wam::internal::Arch;
    using wam::test::require;
    using wam::test::require_error;

    require(wam::internal::arch_name(Arch::unknown) == "unknown",
            "unknown architecture name changed");
    require(wam::internal::arch_name(Arch::gwp05) == "gwp05",
            "GWP05 architecture name changed");
    require(wam::internal::arch_name(Arch::fastwam) == "fastwam",
            "FastWAM architecture name changed");
    require(wam::internal::arch_from_name("gwp05") == Arch::gwp05,
            "GWP05 architecture lookup failed");
    require(wam::internal::arch_from_name("fastwam") == Arch::fastwam,
            "FastWAM architecture lookup failed");
    require(wam::internal::arch_from_name("GWP05") == Arch::unknown,
            "architecture names must be strict and case-sensitive");
    require(wam::internal::arch_from_name("pi05") == Arch::unknown,
            "unknown architecture alias must not be accepted");

    wam::internal::ModelRegistry registry;
    require(registry.find(Arch::gwp05) == nullptr,
            "new registry must be empty");
    registry.add(Arch::gwp05, unsupported_factory());
    require(registry.find(Arch::gwp05) != nullptr,
            "registered architecture was not found");

    require_error([&] { registry.add(Arch::unknown, unsupported_factory()); },
                  wam::ErrorCode::invalid_argument,
                  "unknown architecture registration");
    require_error([&] { registry.add(Arch::fastwam, {}); },
                  wam::ErrorCode::invalid_argument,
                  "empty factory registration");
    require_error([&] { registry.add(Arch::gwp05, unsupported_factory()); },
                  wam::ErrorCode::failed_precondition,
                  "duplicate architecture registration");

    verify_builtin(Arch::gwp05, WAM_BUILD_GWP05 != 0);
    verify_builtin(Arch::fastwam, WAM_BUILD_FASTWAM != 0);
    return 0;
}
