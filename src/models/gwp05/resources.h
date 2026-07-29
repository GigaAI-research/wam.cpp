#pragma once

#include "models/gwp05/state.h"

namespace wam::internal {
class GgufReader;
}

namespace wam::internal::gwp05 {

void refresh_component_telemetry(ModelResources & resources);
bool component_is_loaded(const ExecutionState & state,
                         WeightComponent component);
void unload_component(ModelResources & resources, WeightComponent component);
bool load_component(GgufReader & reader, ModelResources & resources,
                    WeightComponent component);
bool load_resident_weights(GgufReader & reader, ModelResources & resources);

} // namespace wam::internal::gwp05
