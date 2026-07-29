#pragma once

#include "policy/policy_spec.h"
#include "wam/observation.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wam::internal::policy {

enum class ImageResamplePrecision {
    u8_intermediate,
    f32_intermediate,
};

struct CpuImage {
    std::vector<float> pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    TensorLayout layout = TensorLayout::chw;
};

void validate_named_images(const std::vector<ImageView> & images,
                           const ImageSpec & spec);
std::vector<std::size_t> resolve_image_order(const std::vector<ImageView> & images,
                                             const ImageSpec & spec);
CpuImage transform_image_reference(const ImageView & image,
                                   const ImageTransformSpec & transform,
                                   const ImageSpec & spec,
                                   ImageResamplePrecision precision =
                                       ImageResamplePrecision::u8_intermediate);
CpuImage compose_canvas_reference(const std::vector<CpuImage> & images,
                                  const ImageSpec & spec);

} // namespace wam::internal::policy
