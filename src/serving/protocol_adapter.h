#pragma once

#include "model_internal.h"

#include <string>

namespace wam::serving {

std::string model_metadata_json(const ModelInfo & info,
                                const internal::policy::PolicySpecDraft & spec);
std::string error_details_json(const std::vector<ErrorDetail> & details);

} // namespace wam::serving
