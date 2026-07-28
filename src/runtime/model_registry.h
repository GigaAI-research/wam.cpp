#pragma once

#include "model_internal.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace wam::internal {

class GgufReader;

using ModelFactory = std::function<std::unique_ptr<ModelImpl>(
    const RuntimeConfig & options,
    ModelInfo info,
    std::optional<policy::PolicySpec> policy_spec,
    std::shared_ptr<GgufReader> reader)>;

struct ArchitectureDescriptor {
    std::string architecture;
    Capabilities capabilities;
    ModelFactory factory;
};

class ModelRegistry {
public:
    void add(ArchitectureDescriptor descriptor);
    const ArchitectureDescriptor * find(
        std::string_view architecture) const;

private:
    std::unordered_map<std::string, ArchitectureDescriptor> descriptors_;
};

ModelRegistry & model_registry();
void register_builtin_models(ModelRegistry & registry);

} // namespace wam::internal
