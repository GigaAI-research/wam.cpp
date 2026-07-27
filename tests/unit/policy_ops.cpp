#include "support/test_utils.h"

#include "policy/action_ops.h"
#include "policy/image_ops.h"
#include "policy/state_ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

wam::TensorView f32_view(const std::vector<float> & values,
                         std::vector<std::int64_t> shape,
                         std::string layout = {}) {
    wam::TensorView view;
    view.data = values.data();
    view.byte_size = values.size() * sizeof(float);
    view.dtype = wam::DType::f32;
    view.shape = std::move(shape);
    view.layout = std::move(layout);
    view.byte_order = wam::ByteOrder::little;
    return view;
}

wam::ImageView image(const std::string & name,
                     const std::vector<std::uint8_t> & pixels,
                     std::uint32_t width, std::uint32_t height) {
    wam::ImageView view;
    view.name = name;
    view.encoding = wam::ImageEncoding::rgb_u8;
    view.data = pixels.data();
    view.byte_size = pixels.size();
    view.width = width;
    view.height = height;
    view.channels = 3;
    return view;
}

bool near(float first, float second, float tolerance = 1.0e-6F) {
    return std::abs(first - second) <= tolerance;
}

} // namespace

int main() {
    using namespace wam::internal::policy;
    using wam::test::require;
    using wam::test::require_error;

    const std::vector<std::uint8_t> scene = {
        0, 0, 0, 255, 0, 0,
    };
    const std::vector<std::uint8_t> wrist = {255, 255, 255};
    const std::vector<wam::ImageView> unordered_images = {
        image("wrist", wrist, 1, 1), image("scene", scene, 2, 1)};
    ImageSpec image_spec;
    image_spec.views = {
        {"scene", 1, 2, ResizeMode::none,
         InterpolationMode::bilinear, true},
        {"wrist", 1, 1, ResizeMode::none,
         InterpolationMode::bilinear, true},
    };
    image_spec.composition.kind = ImageCompositionKind::canvas;
    image_spec.composition.width = 2;
    image_spec.composition.height = 2;
    image_spec.composition.placements = {
        {"scene", 0, 0, 2, 1}, {"wrist", 1, 1, 1, 1}};
    image_spec.pixel_range = PixelRange::minus_one_to_one;
    image_spec.tensor_layout = TensorLayout::chw;

    const std::vector<std::size_t> order =
        resolve_image_order(unordered_images, image_spec);
    require(order == std::vector<std::size_t>({1, 0}),
            "named image ordering changed");
    std::vector<CpuImage> transformed;
    for (std::size_t index = 0; index < order.size(); ++index) {
        transformed.push_back(transform_image_reference(
            unordered_images[order[index]], image_spec.views[index],
            image_spec));
    }
    const CpuImage canvas = compose_canvas_reference(transformed, image_spec);
    require(canvas.width == 2 && canvas.height == 2 && canvas.channels == 3 &&
                canvas.layout == TensorLayout::chw,
            "canvas geometry changed");
    const std::vector<float> expected_canvas = {
        -1.0F, 1.0F, -1.0F, 1.0F,
        -1.0F, -1.0F, -1.0F, 1.0F,
        -1.0F, -1.0F, -1.0F, 1.0F,
    };
    require(canvas.pixels == expected_canvas,
            "canvas composition, pixel range, or CHW layout changed");

    const std::vector<std::uint8_t> crop_source = {
        0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255,
        0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255,
    };
    ImageSpec crop_spec;
    crop_spec.pixel_range = PixelRange::zero_to_one;
    crop_spec.tensor_layout = TensorLayout::hwc;
    const ImageTransformSpec crop_transform{
        "crop", 2, 2, ResizeMode::cover_center_crop,
        InterpolationMode::bilinear, true};
    const CpuImage cropped = transform_image_reference(
        image("crop", crop_source, 4, 2), crop_transform, crop_spec);
    require(near(cropped.pixels[0], 64.0F / 255.0F) &&
                near(cropped.pixels[3], 128.0F / 255.0F) &&
                near(cropped.pixels[6], 64.0F / 255.0F) &&
                near(cropped.pixels[9], 128.0F / 255.0F),
            "cover-center-crop reference changed");

    const std::vector<std::uint8_t> stretch_source = {
        0, 0, 0, 255, 255, 255};
    const ImageTransformSpec stretch_transform{
        "stretch", 1, 1, ResizeMode::stretch,
        InterpolationMode::bilinear, true};
    const CpuImage stretched = transform_image_reference(
        image("stretch", stretch_source, 2, 1), stretch_transform,
        crop_spec);
    require(stretched.pixels.size() == 3 &&
                near(stretched.pixels[0], 128.0F / 255.0F) &&
                near(stretched.pixels[1], 128.0F / 255.0F) &&
                near(stretched.pixels[2], 128.0F / 255.0F),
            "stretch antialias reference changed");

    const std::vector<std::uint8_t> pil_source = {
        13, 50, 87, 124, 161, 198, 235, 16, 53, 90, 127, 164,
        201, 238, 19, 56, 93, 130, 167, 204, 241, 22, 59, 96,
        133, 170, 207, 244, 25, 62, 99, 136, 173, 210, 247, 28,
        65, 102, 139, 176, 213, 250, 31, 68, 105, 142, 179, 216,
        253, 34, 71, 108, 145, 182, 219, 0, 37, 74, 111, 148,
    };
    const std::vector<std::uint8_t> pil_11_3_reference = {
        92, 129, 150, 128, 104, 133, 163, 137, 112,
        159, 149, 139, 134, 122, 135, 126, 95, 132,
    };
    const ImageTransformSpec pil_transform{
        "pil", 2, 3, ResizeMode::stretch,
        InterpolationMode::bilinear, true};
    const CpuImage pil_resized = transform_image_reference(
        image("pil", pil_source, 5, 4), pil_transform, crop_spec);
    require(pil_resized.pixels.size() == pil_11_3_reference.size(),
            "PIL resize fixture geometry changed");
    for (std::size_t i = 0; i < pil_resized.pixels.size(); ++i) {
        require(static_cast<int>(std::lround(pil_resized.pixels[i] * 255.0F)) ==
                    pil_11_3_reference[i],
                "PIL bilinear uint8 parity changed");
    }

    const std::vector<std::uint8_t> edge_source = {
        0, 0, 0, 64, 64, 64, 128, 128, 128,
        192, 192, 192, 255, 255, 255,
    };
    const ImageTransformSpec edge_transform{
        "edge", 2, 1, ResizeMode::stretch,
        InterpolationMode::bilinear, true};
    const CpuImage edge_resized = transform_image_reference(
        image("edge", edge_source, 1, 5), edge_transform, crop_spec);
    require(edge_resized.pixels.size() == 6 &&
                near(edge_resized.pixels[0], 64.0F / 255.0F) &&
                near(edge_resized.pixels[1], 64.0F / 255.0F) &&
                near(edge_resized.pixels[2], 64.0F / 255.0F),
            "antialias boundary support must be truncated and renormalized");
    ImageSpec legacy_edge_spec = crop_spec;
    legacy_edge_spec.resample_boundary = ResampleBoundaryMode::clamp;
    const CpuImage legacy_edge_resized = transform_image_reference(
        image("edge", edge_source, 1, 5), edge_transform, legacy_edge_spec);
    require(near(legacy_edge_resized.pixels[0], 56.0F / 255.0F),
            "legacy antialias boundary clamp changed");

    StateSpec state_spec;
    state_spec.real_dim = 2;
    state_spec.model_dim = 4;
    state_spec.pad_value = 7.0F;
    state_spec.normalization = {NormalizationKind::quantile, false, 1.0e-8F};
    state_spec.stats.lower = {0.0F, 0.0F, 0.0F, 0.0F};
    state_spec.stats.upper = {10.0F, 10.0F, 1.0F, 1.0F};
    state_spec.stats.mask = {1, 1, 0, 0};
    const std::vector<float> raw_state_values = {0.0F, 5.0F};
    const wam::TensorView raw_state =
        f32_view(raw_state_values, {2}, "D");
    validate_state_input(raw_state, state_spec);
    const std::vector<float> copied_state = read_state_f32(raw_state);
    std::vector<float> model_state = pad_state(copied_state, state_spec);
    normalize_state_reference(model_state, state_spec, state_spec.stats);
    require(model_state == std::vector<float>({-1.0F, 0.0F, 7.0F, 7.0F}),
            "state padding, mask, or quantile normalization changed");

    StateSpec z_state_spec;
    z_state_spec.real_dim = 2;
    z_state_spec.model_dim = 2;
    z_state_spec.normalization =
        {NormalizationKind::z_score, false, 1.0e-6F};
    z_state_spec.stats.mean = {1.0F, 2.0F};
    z_state_spec.stats.stddev = {2.0F, 4.0F};
    std::vector<float> z_state = {3.0F, 6.0F};
    normalize_state_reference(z_state, z_state_spec, z_state_spec.stats);
    require(z_state == std::vector<float>({1.0F, 1.0F}),
            "z-score state normalization changed");
    z_state_spec.normalization.output_clamp_lower = -5.0F;
    z_state_spec.normalization.output_clamp_upper = 5.0F;
    z_state = {101.0F, -98.0F};
    normalize_state_reference(z_state, z_state_spec, z_state_spec.stats);
    require(z_state == std::vector<float>({5.0F, -5.0F}),
            "z-score state output clamp changed");

    ActionSpec action_spec;
    action_spec.horizon = 2;
    action_spec.real_dim = 2;
    action_spec.model_dim = 3;
    action_spec.normalization =
        {NormalizationKind::quantile, false, 1.0e-8F};
    action_spec.stats.lower = {0.0F, 10.0F, 0.0F};
    action_spec.stats.upper = {2.0F, 20.0F, 1.0F};
    action_spec.stats.mask = {1, 1, 0};
    action_spec.recovery.kind = ActionRecoveryKind::add_current_state;
    action_spec.recovery.reference_state_indices = {0, -1};
    const PolicyActionChunk decoded = decode_action_reference(
        {-1.0F, 0.0F, 99.0F, 1.0F, 1.0F, 98.0F},
        {3.0F, 4.0F}, action_spec, action_spec.stats);
    require(decoded.horizon == 2 && decoded.action_dim == 2 &&
                decoded.values ==
                    std::vector<float>({3.0F, 15.0F, 5.0F, 20.0F}),
            "action trim, unnormalize, or raw-state recovery changed");
    const wam::Tensor action_tensor = make_action_tensor(decoded);
    require(action_tensor.shape == std::vector<std::int64_t>({2, 2}) &&
                action_tensor.layout == "T,A" &&
                action_tensor.byte_order == wam::ByteOrder::little,
            "public action tensor contract changed");

    ActionSpec z_action_spec;
    z_action_spec.horizon = 1;
    z_action_spec.real_dim = 2;
    z_action_spec.model_dim = 2;
    z_action_spec.normalization =
        {NormalizationKind::z_score, false, 1.0e-6F};
    z_action_spec.normalization.output_clamp_lower = -1.0F;
    z_action_spec.normalization.output_clamp_upper = 1.0F;
    z_action_spec.stats.mean = {10.0F, -2.0F};
    z_action_spec.stats.stddev = {2.0F, 0.5F};
    z_action_spec.recovery.kind = ActionRecoveryKind::identity;
    const PolicyActionChunk z_decoded = decode_action_reference(
        {1.0F, 2.0F}, {}, z_action_spec, z_action_spec.stats);
    require(z_decoded.values == std::vector<float>({12.0F, -1.5F}),
            "action output clamp or z-score unnormalization changed");

    const std::vector<float> explicit_values(6, 0.25F);
    const wam::TensorView explicit_noise =
        f32_view(explicit_values, {2, 3}, "T,A");
    std::mt19937 with_explicit(17);
    require(prepare_action_noise(explicit_noise, action_spec, with_explicit) ==
                explicit_values,
            "explicit action noise changed");
    const std::vector<float> generated_after_explicit =
        prepare_action_noise({}, action_spec, with_explicit);
    std::mt19937 reference(17);
    const std::vector<float> first_generated =
        prepare_action_noise({}, action_spec, reference);
    require(generated_after_explicit == first_generated,
            "explicit action noise advanced the session RNG");
    const std::vector<float> second_generated =
        prepare_action_noise({}, action_spec, reference);
    require(second_generated != first_generated,
            "consecutive generated action noise did not advance RNG");
    reference.seed(17);
    require(prepare_action_noise({}, action_spec, reference) == first_generated,
            "action noise reset did not restore the initial seed");

    require_error(
        [&] {
            wam::TensorView bad = explicit_noise;
            bad.shape = {1, 2, 3};
            (void) prepare_action_noise(bad, action_spec, reference);
        },
        wam::ErrorCode::invalid_argument,
        "rank-three action noise must be rejected");
    require_error(
        [&] {
            wam::TensorView partial;
            partial.shape = {2, 3};
            partial.dtype = wam::DType::f32;
            partial.byte_order = wam::ByteOrder::little;
            (void) prepare_action_noise(partial, action_spec, reference);
        },
        wam::ErrorCode::invalid_argument,
        "partially initialized action noise must not mean omitted");
    require_error(
        [&] {
            std::vector<wam::ImageView> duplicate = unordered_images;
            duplicate[0].name = "scene";
            validate_named_images(duplicate, image_spec);
        },
        wam::ErrorCode::invalid_argument,
        "duplicate image role must be rejected");
    return 0;
}
