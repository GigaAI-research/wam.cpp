#pragma once

#include "arch.h"
#include "model_internal.h"

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

namespace wam::internal {

class GgufReader;

using ModelFactory = std::function<std::unique_ptr<ModelImpl>(
    const RuntimeConfig & options,
    ModelInfo info,
    std::optional<policy::PolicySpec> policy_spec,
    std::shared_ptr<GgufReader> reader)>;

class ModelRegistry {
public:
    void add(Arch arch, ModelFactory factory);
    const ModelFactory * find(Arch arch) const noexcept;

private:
    struct ArchHash {
        std::size_t operator()(Arch arch) const noexcept;
    };

    std::unordered_map<Arch, ModelFactory, ArchHash> factories_;
};

ModelRegistry & model_registry();
void register_builtin_models(ModelRegistry & registry);

} // namespace wam::internal
