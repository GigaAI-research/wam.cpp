#include "model_registry.h"

#include "wam/types.h"

#if WAM_BUILD_GWP05
#include "models/gwp05/model.h"
#endif

#include <utility>

namespace wam::internal {

void ModelRegistry::add(Arch arch, ModelFactory factory) {
    if (arch == Arch::unknown || !factory) {
        throw Error(ErrorCode::internal, "invalid model factory registration");
    }
    const auto result = entries_.emplace(
        arch, RegistryEntry{arch, std::move(factory)});
    if (!result.second) {
        throw Error(ErrorCode::internal, "duplicate model factory registration");
    }
}

const RegistryEntry * ModelRegistry::find(Arch arch) const noexcept {
    const auto iterator = entries_.find(arch);
    return iterator == entries_.end() ? nullptr : &iterator->second;
}

void register_builtin_models(ModelRegistry & registry) {
#if WAM_BUILD_GWP05
    register_gwp05(registry);
#else
    (void) registry;
#endif
}

ModelRegistry & model_registry() {
    static ModelRegistry registry = [] {
        ModelRegistry value;
        register_builtin_models(value);
        return value;
    }();
    return registry;
}

} // namespace wam::internal
