#include "runtime/model_registry.h"

#include "wam/error.h"

#include <utility>

namespace wam::internal {

void ModelRegistry::add(ArchitectureDescriptor descriptor) {
    if (descriptor.architecture.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "model architecture id must not be empty");
    }
    if (!descriptor.factory) {
        throw Error(ErrorCode::invalid_argument,
                    "cannot register an empty model factory");
    }

    const std::string architecture = descriptor.architecture;
    const auto inserted = descriptors_.emplace(
        architecture, std::move(descriptor));
    if (!inserted.second) {
        throw Error(ErrorCode::failed_precondition,
                    "model architecture is already registered: " +
                        architecture);
    }
}

const ArchitectureDescriptor * ModelRegistry::find(
    std::string_view architecture) const {
    const auto it = descriptors_.find(std::string(architecture));
    return it == descriptors_.end() ? nullptr : &it->second;
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
