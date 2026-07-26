#include "model_registry.h"

#include "models/fastwam/model.h"
#include "models/gwp05/model.h"

#include <utility>

namespace wam::internal {

void ModelRegistry::add(Arch arch, ModelFactory factory) {
    if (arch == Arch::unknown) {
        throw Error(ErrorCode::invalid_argument,
                    "cannot register an unknown model architecture");
    }
    if (!factory) {
        throw Error(ErrorCode::invalid_argument,
                    "cannot register an empty model factory");
    }

    const auto inserted = factories_.emplace(arch, std::move(factory));
    if (!inserted.second) {
        throw Error(ErrorCode::failed_precondition,
                    "model architecture is already registered: " +
                        std::string(arch_name(arch)));
    }
}

const ModelFactory * ModelRegistry::find(Arch arch) const noexcept {
    const auto it = factories_.find(arch);
    return it == factories_.end() ? nullptr : &it->second;
}

std::size_t ModelRegistry::ArchHash::operator()(Arch arch) const noexcept {
    return static_cast<std::size_t>(arch);
}

ModelRegistry & model_registry() {
    static ModelRegistry registry = [] {
        ModelRegistry value;
        register_builtin_models(value);
        return value;
    }();
    return registry;
}

void register_builtin_models(ModelRegistry & registry) {
    (void) registry;
#if WAM_BUILD_GWP05
    register_gwp05(registry);
#endif
#if WAM_BUILD_FASTWAM
    register_fastwam(registry);
#endif
}

} // namespace wam::internal
