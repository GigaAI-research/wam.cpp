#include "models/gwp05/module.h"

#include "models/gwp05/model.h"

namespace wam::internal::gwp05 {

ArchitectureDescriptor module_descriptor() {
    ArchitectureDescriptor descriptor;
    descriptor.architecture = "gwp05";
    descriptor.capabilities.action = true;
    descriptor.capabilities.raw_images = true;
    descriptor.capabilities.token_input = true;
    descriptor.capabilities.precomputed_embedding = true;
    descriptor.capabilities.explicit_action_noise = true;
    descriptor.factory = create_model;
    return descriptor;
}

} // namespace wam::internal::gwp05
