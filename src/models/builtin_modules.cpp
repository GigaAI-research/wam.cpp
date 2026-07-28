#include "model_registry.h"

#if WAM_BUILD_FASTWAM
#include "models/fastwam/model.h"
#endif
#if WAM_BUILD_GWP05
#include "models/gwp05/model.h"
#endif

namespace wam::internal {

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
