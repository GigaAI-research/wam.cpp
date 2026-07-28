#include "policy/image_ops.h"

#include "wam/error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

namespace wam::internal::policy {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

std::size_t image_elements(std::uint32_t width, std::uint32_t height,
                           std::uint32_t channels,
                           const std::string & field) {
    if (width == 0 || height == 0 || channels == 0 ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        static_cast<std::size_t>(width) * height >
            std::numeric_limits<std::size_t>::max() / channels) {
        invalid("image element count overflows", field, "invalid geometry");
    }
    return static_cast<std::size_t>(width) * height * channels;
}

std::size_t image_index(const CpuImage & image, std::uint32_t x,
                        std::uint32_t y, std::uint32_t channel) {
    if (image.layout == TensorLayout::hwc) {
        return (static_cast<std::size_t>(y) * image.width + x) *
                   image.channels + channel;
    }
    return (static_cast<std::size_t>(channel) * image.height + y) *
               image.width + x;
}

std::uint8_t source_value(const ImageView & image, std::uint32_t x,
                          std::uint32_t y, std::uint32_t channel) {
    const std::size_t packed_stride =
        static_cast<std::size_t>(image.width) * image.channels;
    const std::size_t stride = image.row_stride_bytes == 0
        ? packed_stride : image.row_stride_bytes;
    return image.data[static_cast<std::size_t>(y) * stride +
                      static_cast<std::size_t>(x) * image.channels + channel];
}

float output_range(float value, PixelRange range) {
    return range == PixelRange::minus_one_to_one
        ? value * 2.0F - 1.0F : value;
}

std::uint32_t round_ties_to_even(float value) {
    const float floor_value = std::floor(value);
    const float fraction = value - floor_value;
    std::uint64_t rounded = static_cast<std::uint64_t>(floor_value);
    if (fraction > 0.5F ||
        (fraction == 0.5F && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    if (rounded == 0 ||
        rounded > std::numeric_limits<std::uint32_t>::max()) {
        invalid("resized image dimension is invalid", "images",
                "zero or overflow");
    }
    return static_cast<std::uint32_t>(rounded);
}

struct AxisSample {
    std::vector<std::uint32_t> indices;
    std::vector<std::int32_t> weights;
};

double cubic_weight(double value) {
    constexpr double a = -0.5;
    const double x = std::abs(value);
    if (x < 1.0) {
        return (a + 2.0) * x * x * x -
               (a + 3.0) * x * x + 1.0;
    }
    if (x < 2.0) {
        return a * x * x * x - 5.0 * a * x * x +
               8.0 * a * x - 4.0 * a;
    }
    return 0.0;
}

AxisSample build_axis_sample(std::uint32_t output_position,
                             std::uint32_t source_size,
                             std::uint32_t resized_size,
                             InterpolationMode interpolation,
                             bool antialias,
                             ResampleBoundaryMode boundary) {
    constexpr std::int32_t coefficient_scale = 1 << 22;
    const double scale = static_cast<double>(resized_size) / source_size;
    const double center = (output_position + 0.5) / scale - 0.5;
    AxisSample sample;
    if (interpolation == InterpolationMode::nearest) {
        const int index = static_cast<int>(std::floor(center + 0.5F));
        sample.indices.push_back(static_cast<std::uint32_t>(
            std::max(0, std::min(static_cast<int>(source_size) - 1, index))));
        sample.weights.push_back(coefficient_scale);
        return sample;
    }

    const double base_radius = interpolation == InterpolationMode::bicubic
        ? 2.0 : 1.0;
    const double filter_scale = antialias
        ? std::max(1.0, 1.0 / scale) : 1.0;
    const double radius = base_radius * filter_scale;
    const int first = static_cast<int>(std::ceil(center - radius));
    const int last = static_cast<int>(std::floor(center + radius));
    std::vector<double> weights;
    double total = 0.0;
    for (int index = first; index <= last; ++index) {
        const double distance = (index - center) / filter_scale;
        const double weight = interpolation == InterpolationMode::bicubic
            ? cubic_weight(distance)
            : std::max(0.0, 1.0 - std::abs(distance));
        if (weight == 0.0) continue;
        if (boundary == ResampleBoundaryMode::truncate &&
            (index < 0 || index >= static_cast<int>(source_size))) {
            continue;
        }
        sample.indices.push_back(static_cast<std::uint32_t>(
            std::max(0, std::min(static_cast<int>(source_size) - 1, index))));
        weights.push_back(weight);
        total += weight;
    }
    if (sample.indices.empty() || total == 0.0) {
        invalid("image interpolation produced an empty sample", "images",
                "invalid resize geometry");
    }
    sample.weights.reserve(weights.size());
    for (const double weight : weights) {
        const double scaled = weight / total * coefficient_scale;
        sample.weights.push_back(static_cast<std::int32_t>(
            scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
    }
    return sample;
}

std::uint8_t resample_u8(const std::uint8_t * values,
                         const AxisSample & sample,
                         std::size_t stride) {
    constexpr int precision_bits = 22;
    std::int64_t sum = std::int64_t{1} << (precision_bits - 1);
    for (std::size_t i = 0; i < sample.indices.size(); ++i) {
        sum += static_cast<std::int64_t>(values[sample.indices[i] * stride]) *
               sample.weights[i];
    }
    const std::int64_t rounded = sum >> precision_bits;
    return static_cast<std::uint8_t>(
        std::max<std::int64_t>(0, std::min<std::int64_t>(255, rounded)));
}

void validate_rgb_image(const ImageView & image) {
    const std::string field = "images." + image.name;
    if (image.encoding != ImageEncoding::rgb_u8) {
        throw Error(ErrorCode::unsupported,
                    "decoded RGB images are required",
                    {{field + ".encoding", "expected rgb_u8"}});
    }
    if (image.name.empty() || image.data == nullptr || image.width == 0 ||
        image.height == 0 || image.channels != 3 ||
        image.width > static_cast<std::uint32_t>(
            std::numeric_limits<int>::max()) ||
        image.height > static_cast<std::uint32_t>(
            std::numeric_limits<int>::max())) {
        invalid("RGB image payload or geometry is invalid", field,
                "expected nonempty named HWC RGB data");
    }
    if (image.width >
        std::numeric_limits<std::size_t>::max() / image.channels) {
        invalid("RGB image row size overflows", field, "overflow");
    }
    const std::size_t packed_stride =
        static_cast<std::size_t>(image.width) * image.channels;
    const std::size_t stride = image.row_stride_bytes == 0
        ? packed_stride : image.row_stride_bytes;
    if (stride < packed_stride ||
        image.height > std::numeric_limits<std::size_t>::max() / stride ||
        image.byte_size != stride * image.height) {
        invalid("RGB image byte size or row stride is invalid", field,
                "shape mismatch");
    }
}

} // namespace

void validate_named_images(const std::vector<ImageView> & images,
                           const ImageSpec & spec) {
    if (images.size() != spec.views.size()) {
        invalid("image count does not match PolicySpec", "images",
                "expected " + std::to_string(spec.views.size()) +
                    ", got " + std::to_string(images.size()));
    }
    std::unordered_map<std::string, std::size_t> counts;
    for (const ImageView & image : images) {
        validate_rgb_image(image);
        ++counts[image.name];
    }
    for (const ImageTransformSpec & view : spec.views) {
        const std::size_t count = counts[view.role];
        if (count == 0) {
            invalid("required named image is missing", "images", view.role);
        }
        if (count != 1) {
            invalid("named image appears more than once", "images",
                    "duplicate " + view.role);
        }
    }
    for (const auto & entry : counts) {
        const auto found = std::find_if(
            spec.views.begin(), spec.views.end(),
            [&](const ImageTransformSpec & view) {
                return view.role == entry.first;
            });
        if (found == spec.views.end()) {
            invalid("image role is not declared by PolicySpec", "images",
                    entry.first);
        }
    }
}

std::vector<std::size_t> resolve_image_order(
    const std::vector<ImageView> & images, const ImageSpec & spec) {
    validate_named_images(images, spec);
    std::vector<std::size_t> order;
    order.reserve(spec.views.size());
    for (const ImageTransformSpec & view : spec.views) {
        const auto found = std::find_if(
            images.begin(), images.end(), [&](const ImageView & image) {
                return image.name == view.role;
            });
        order.push_back(static_cast<std::size_t>(found - images.begin()));
    }
    return order;
}

CpuImage transform_image_reference(const ImageView & image,
                                   const ImageTransformSpec & transform,
                                   const ImageSpec & spec) {
    validate_rgb_image(image);
    if (image.name != transform.role) {
        invalid("image role does not match transform", "images." + image.name,
                "expected " + transform.role);
    }
    if (transform.target_width == 0 || transform.target_height == 0) {
        invalid("image transform has zero target geometry", "images." + image.name,
                "zero dimension");
    }
    if (transform.target_width > static_cast<std::uint32_t>(
            std::numeric_limits<int>::max()) ||
        transform.target_height > static_cast<std::uint32_t>(
            std::numeric_limits<int>::max())) {
        invalid("image transform target is too large", "images." + image.name,
                "geometry exceeds reference implementation limits");
    }
    if (transform.resize == ResizeMode::none &&
        (image.width != transform.target_width ||
         image.height != transform.target_height)) {
        invalid("image dimensions do not match resize=none", "images." + image.name,
                "shape mismatch");
    }

    std::uint32_t resized_width = transform.target_width;
    std::uint32_t resized_height = transform.target_height;
    std::uint32_t crop_x = 0;
    std::uint32_t crop_y = 0;
    if (transform.resize == ResizeMode::cover_center_crop) {
        const float scale = std::max(
            static_cast<float>(transform.target_width) / image.width,
            static_cast<float>(transform.target_height) / image.height);
        resized_width = round_ties_to_even(image.width * scale);
        resized_height = round_ties_to_even(image.height * scale);
        crop_x = (resized_width - transform.target_width) / 2;
        crop_y = (resized_height - transform.target_height) / 2;
    }
    if (transform.resize == ResizeMode::none) {
        resized_width = image.width;
        resized_height = image.height;
    }

    std::vector<AxisSample> x_plan(transform.target_width);
    std::vector<AxisSample> y_plan(transform.target_height);
    for (std::uint32_t x = 0; x < transform.target_width; ++x) {
        x_plan[x] = build_axis_sample(x + crop_x, image.width, resized_width,
                                     transform.interpolation,
                                     transform.antialias,
                                     spec.resample_boundary);
    }
    for (std::uint32_t y = 0; y < transform.target_height; ++y) {
        y_plan[y] = build_axis_sample(y + crop_y, image.height, resized_height,
                                     transform.interpolation,
                                     transform.antialias,
                                     spec.resample_boundary);
    }

    std::vector<std::uint8_t> horizontal(image_elements(
        transform.target_width, image.height, 3, "images." + image.name));
    std::vector<std::uint8_t> row(image.width);
    for (std::uint32_t y = 0; y < image.height; ++y) {
        for (std::uint32_t channel = 0; channel < 3; ++channel) {
            for (std::uint32_t source_x = 0; source_x < image.width;
                 ++source_x) {
                row[source_x] = source_value(image, source_x, y, channel);
            }
            for (std::uint32_t x = 0; x < transform.target_width; ++x) {
                horizontal[(static_cast<std::size_t>(y) *
                            transform.target_width + x) * 3 + channel] =
                    resample_u8(row.data(), x_plan[x], 1);
            }
        }
    }

    CpuImage output;
    output.width = transform.target_width;
    output.height = transform.target_height;
    output.channels = 3;
    output.layout = spec.tensor_layout;
    output.pixels.resize(image_elements(
        output.width, output.height, output.channels,
        "images." + image.name));
    for (std::uint32_t y = 0; y < output.height; ++y) {
        for (std::uint32_t x = 0; x < output.width; ++x) {
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const std::uint8_t value = resample_u8(
                    horizontal.data() +
                        (static_cast<std::size_t>(x) * 3 + channel),
                    y_plan[y], static_cast<std::size_t>(output.width) * 3);
                output.pixels[image_index(output, x, y, channel)] =
                    output_range(value * (1.0F / 255.0F), spec.pixel_range);
            }
        }
    }
    return output;
}

CpuImage compose_canvas_reference(const std::vector<CpuImage> & images,
                                  const ImageSpec & spec) {
    if (spec.composition.kind != ImageCompositionKind::canvas ||
        images.size() != spec.views.size() ||
        spec.composition.placements.size() != spec.views.size()) {
        invalid("canvas image composition contract is invalid",
                "images.composition", "expected one image per placement");
    }
    CpuImage canvas;
    canvas.width = spec.composition.width;
    canvas.height = spec.composition.height;
    canvas.channels = 3;
    canvas.layout = spec.tensor_layout;
    const float background = spec.pixel_range == PixelRange::minus_one_to_one
        ? -1.0F : 0.0F;
    canvas.pixels.assign(image_elements(
        canvas.width, canvas.height, canvas.channels,
        "images.composition"), background);

    for (std::size_t i = 0; i < spec.views.size(); ++i) {
        const ImageTransformSpec & view = spec.views[i];
        const auto placement = std::find_if(
            spec.composition.placements.begin(),
            spec.composition.placements.end(),
            [&](const ImagePlacement & candidate) {
                return candidate.role == view.role;
            });
        if (placement == spec.composition.placements.end()) {
            invalid("canvas placement is missing", "images.composition",
                    view.role);
        }
        const CpuImage & image = images[i];
        if (image.width != placement->width ||
            image.height != placement->height || image.channels != 3 ||
            image.layout != canvas.layout) {
            invalid("transformed image does not match canvas placement",
                    "images." + view.role, "shape or layout mismatch");
        }
        for (std::uint32_t y = 0; y < image.height; ++y) {
            for (std::uint32_t x = 0; x < image.width; ++x) {
                for (std::uint32_t channel = 0; channel < 3; ++channel) {
                    canvas.pixels[image_index(
                        canvas, placement->x + x, placement->y + y, channel)] =
                        image.pixels[image_index(image, x, y, channel)];
                }
            }
        }
    }
    return canvas;
}

} // namespace wam::internal::policy
