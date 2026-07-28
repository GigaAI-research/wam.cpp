#include "runtime/telemetry.h"

#include <string>

namespace wam::internal::runtime {

void append_timing(Telemetry & telemetry, std::string_view name,
                   double milliseconds) {
    if (milliseconds <= 0.0) return;
    telemetry.model_timings.push_back({std::string(name), milliseconds});
}

} // namespace wam::internal::runtime
