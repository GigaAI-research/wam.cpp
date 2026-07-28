#pragma once

#include "wam/error.h"
#include "wam/model.h"
#include "wam/policy_spec.h"

#include <string>
#include <vector>

namespace wam::serving {

std::string model_metadata_json(const ModelInfo & info,
                                const PolicySpec & spec);
std::string error_details_json(const std::vector<ErrorDetail> & details);

} // namespace wam::serving
