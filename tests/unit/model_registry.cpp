#include "support/test_utils.h"

#include "runtime/model_registry.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

wam::internal::ModelFactory unsupported_factory() {
    return [](const wam::RuntimeConfig &, wam::ModelInfo,
              std::optional<wam::internal::policy::PolicySpec>,
              std::shared_ptr<wam::internal::GgufReader>)
               -> std::unique_ptr<wam::internal::ModelImpl> {
        throw wam::Error(wam::ErrorCode::unsupported, "test factory");
    };
}

void verify_builtin(const std::string & architecture, bool expected) {
    const wam::internal::ArchitectureDescriptor * descriptor =
        wam::internal::model_registry().find(architecture);
    wam::test::require((descriptor != nullptr) == expected,
                       "built-in registry configuration mismatch for " +
                           architecture);
    if (descriptor != nullptr && architecture == "fastwam") {
        wam::test::require_error(
            [&] {
                descriptor->factory(wam::RuntimeConfig{}, wam::ModelInfo{},
                                    std::nullopt, nullptr);
            },
            wam::ErrorCode::incompatible_artifact,
            "FastWAM factory must reject a missing artifact contract");
    }
}

} // namespace

int main() {
    using wam::internal::ArchitectureDescriptor;
    using wam::test::require;
    using wam::test::require_error;

    wam::internal::ModelRegistry registry;
    require(registry.find("third-model") == nullptr,
            "new registry must be empty");
    ArchitectureDescriptor third;
    third.architecture = "third-model";
    third.capabilities.action = true;
    third.factory = unsupported_factory();
    registry.add(std::move(third));
    const ArchitectureDescriptor * registered =
        registry.find("third-model");
    require(registered != nullptr && registered->capabilities.action,
            "third model descriptor was not registered");
    require(registry.find("THIRD-MODEL") == nullptr,
            "architecture ids must be strict and case-sensitive");

    require_error([&] { registry.add({}); },
                  wam::ErrorCode::invalid_argument,
                  "empty architecture registration");
    require_error(
        [&] {
            ArchitectureDescriptor empty_factory;
            empty_factory.architecture = "empty-factory";
            registry.add(std::move(empty_factory));
        },
                  wam::ErrorCode::invalid_argument,
                  "empty factory registration");
    require_error(
        [&] {
            ArchitectureDescriptor duplicate;
            duplicate.architecture = "third-model";
            duplicate.factory = unsupported_factory();
            registry.add(std::move(duplicate));
        },
                  wam::ErrorCode::failed_precondition,
                  "duplicate architecture registration");

    verify_builtin("gwp05", WAM_BUILD_GWP05 != 0);
    verify_builtin("fastwam", WAM_BUILD_FASTWAM != 0);
    return 0;
}
