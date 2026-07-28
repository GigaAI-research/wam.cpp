#include "models/fastwam/module.h"

#include "models/fastwam/model.h"

namespace wam::internal::fastwam {

ArchitectureDescriptor module_descriptor() {
    ArchitectureDescriptor descriptor;
    descriptor.architecture = "fastwam";
    descriptor.capabilities.action = true;
    descriptor.capabilities.raw_images = true;
    descriptor.capabilities.precomputed_embedding = true;
    descriptor.capabilities.explicit_action_noise = true;
    descriptor.factory = create_model;
    return descriptor;
}

} // namespace wam::internal::fastwam
