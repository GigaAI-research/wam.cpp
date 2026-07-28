#include "runtime/model_registry.h"

#if WAM_BUILD_FASTWAM
#include "models/fastwam/module.h"
#endif
#if WAM_BUILD_GWP05
#include "models/gwp05/module.h"
#endif

namespace wam::internal {

void register_builtin_models(ModelRegistry & registry) {
    (void) registry;
#if WAM_BUILD_GWP05
    registry.add(gwp05::module_descriptor());
#endif
#if WAM_BUILD_FASTWAM
    registry.add(fastwam::module_descriptor());
#endif
}

} // namespace wam::internal
