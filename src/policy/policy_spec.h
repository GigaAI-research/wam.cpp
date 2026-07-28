#pragma once

#include "wam/policy_spec.h"

#include <cstddef>
#include <optional>
#include <string>

namespace wam::internal {
class GgufReader;
}

namespace wam::internal::policy {

using ::wam::ActionFrame;
using ::wam::ActionRecoveryKind;
using ::wam::ActionRecoverySpec;
using ::wam::ActionRepresentation;
using ::wam::ActionSpec;
using ::wam::ColorSpace;
using ::wam::GripperEncoding;
using ::wam::ImageCompositionKind;
using ::wam::ImageCompositionSpec;
using ::wam::ImagePlacement;
using ::wam::ImageSpec;
using ::wam::ImageTransformSpec;
using ::wam::InterpolationMode;
using ::wam::LanguageInputMode;
using ::wam::LanguageSpec;
using ::wam::NormalizationKind;
using ::wam::NormalizationSpec;
using ::wam::NormalizationStats;
using ::wam::PixelRange;
using ::wam::PolicyIdentity;
using ::wam::PolicySpec;
using ::wam::ResampleBoundaryMode;
using ::wam::ResizeMode;
using ::wam::SequenceSide;
using ::wam::StateSpec;
using ::wam::TensorLayout;
inline constexpr std::uint32_t kPolicySpecSchemaVersion =
    ::wam::kPolicySpecSchemaVersion;

std::optional<PolicySpec> try_read_policy_spec(const GgufReader & reader);
void validate_policy_spec(const PolicySpec & spec);
std::size_t policy_image_count(const PolicySpec & spec) noexcept;
const ImageTransformSpec & require_image_spec(const PolicySpec & spec,
                                              const std::string & role);

} // namespace wam::internal::policy
