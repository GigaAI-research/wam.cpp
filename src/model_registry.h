#pragma once

#include "arch.h"
#include "model_internal.h"

#include <functional>
#include <memory>
#include <unordered_map>

namespace wam::internal {

class GgufReader;

using ModelFactory = std::function<std::unique_ptr<ModelImpl>(
    const ModelOptions &, ModelInfo, std::shared_ptr<GgufReader>)>;

struct RegistryEntry {
    Arch arch = Arch::unknown;
    ModelFactory factory;
};

class ModelRegistry {
public:
    void add(Arch arch, ModelFactory factory);
    const RegistryEntry * find(Arch arch) const noexcept;

private:
    struct ArchHash {
        std::size_t operator()(Arch value) const noexcept {
            return static_cast<std::size_t>(value);
        }
    };
    std::unordered_map<Arch, RegistryEntry, ArchHash> entries_;
};

ModelRegistry & model_registry();
void register_builtin_models(ModelRegistry & registry);

} // namespace wam::internal
