#include "wam/runtime_config.h"

int main() {
    wam::RuntimeConfig config;
    config.tuning.prefix_cache = false;
    config.tuning.graph_cache = false;
    config.debug_dump.audit_graph_dtypes = true;
    return config.device_index;
}
