#pragma once

#include "wam/prediction.h"

#include <string_view>

namespace wam::internal::runtime {

void append_timing(Telemetry & telemetry, std::string_view name,
                   double milliseconds);

} // namespace wam::internal::runtime
