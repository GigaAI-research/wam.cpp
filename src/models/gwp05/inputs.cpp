#include "inputs.h"

#include "semantics.h"

#include "ggml.h"

#define STBI_FAILURE_USERMSG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace wam::internal::gwp05 {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

bool latent_shape_matches(const std::vector<std::int64_t> & shape,
                          const ArtifactContract & artifact) {
    const auto channels = static_cast<std::int64_t>(artifact.geometry.vae_z_dim);
    const auto height = artifact.sequence_geometry.latent_height;
    const auto width = artifact.sequence_geometry.latent_width;
    return shape == std::vector<std::int64_t>{channels, height, width} ||
        shape == std::vector<std::int64_t>{1, channels, height, width};
}

void prepare_latent(const NamedTensorView & value, const ArtifactContract & artifact,
                    PreparedInputs & prepared) {
    const TensorView & tensor = value.tensor;
    const std::size_t elements = static_cast<std::size_t>(artifact.geometry.vae_z_dim) *
        static_cast<std::size_t>(artifact.sequence_geometry.latent_height) *
        static_cast<std::size_t>(artifact.sequence_geometry.latent_width);
    if (tensor.data == nullptr || tensor.dtype != DType::f32 ||
        tensor.byte_order != ByteOrder::little || !latent_shape_matches(tensor.shape, artifact) ||
        tensor.byte_size != elements * sizeof(float)) {
        invalid("precomputed GWP reference latent has the wrong tensor contract",
                kGwp05ReferenceLatent, "expected little-endian F32 [C,H,W]");
    }
    const float * source = static_cast<const float *>(tensor.data);
    prepared.reference_latent.assign(source, source + elements);
    for (std::size_t index = 0; index < elements; ++index) {
        if (!std::isfinite(prepared.reference_latent[index])) {
            invalid("precomputed GWP reference latent must be finite",
                    kGwp05ReferenceLatent, std::to_string(index));
        }
    }
    prepared.view.precomputed_ref_latent = prepared.reference_latent.data();
    prepared.view.precomputed_ref_latent_n = static_cast<int>(prepared.reference_latent.size());
}

void decode_image(const wam::ImageView & source, std::vector<std::uint8_t> & destination,
                  engine::ImageView & view) {
    if (source.data == nullptr || source.byte_size == 0) {
        invalid("camera image payload is required", source.name, "empty payload");
    }
    if (source.encoding == ImageEncoding::rgb_u8) {
        if (source.width == 0 || source.height == 0 || source.channels != 3) {
            invalid("RGB camera image has invalid geometry", source.name,
                    "expected nonzero HWC with three channels");
        }
        const std::size_t packed_stride = static_cast<std::size_t>(source.width) * 3;
        const std::size_t stride = source.row_stride_bytes == 0
            ? packed_stride : source.row_stride_bytes;
        if (stride < packed_stride || source.byte_size != stride * source.height) {
            invalid("RGB camera byte size or stride is invalid", source.name, "shape mismatch");
        }
        destination.resize(packed_stride * source.height);
        for (std::size_t row = 0; row < source.height; ++row) {
            std::memcpy(destination.data() + row * packed_stride,
                        source.data + row * stride, packed_stride);
        }
        view = {destination.data(), static_cast<int>(source.width),
                static_cast<int>(source.height), engine::PixelFormat::U8};
        return;
    }
    if (source.encoding != ImageEncoding::png && source.encoding != ImageEncoding::jpeg) {
        invalid("unsupported camera image encoding", source.name,
                std::to_string(static_cast<std::uint32_t>(source.encoding)));
    }
    if (source.byte_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        invalid("encoded camera image is too large", source.name, "payload exceeds decoder limit");
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc * decoded = stbi_load_from_memory(
        source.data, static_cast<int>(source.byte_size), &width, &height, &channels, 3);
    if (decoded == nullptr || width <= 0 || height <= 0) {
        const char * reason = stbi_failure_reason();
        invalid("cannot decode camera image", source.name, reason == nullptr ? "decode failed" : reason);
    }
    if ((source.width != 0 && source.width != static_cast<std::uint32_t>(width)) ||
        (source.height != 0 && source.height != static_cast<std::uint32_t>(height)) ||
        (source.channels != 0 && source.channels != 3)) {
        stbi_image_free(decoded);
        invalid("encoded camera metadata disagrees with decoded geometry", source.name,
                "width, height, or channels mismatch");
    }
    destination.assign(decoded, decoded + static_cast<std::size_t>(width) * height * 3);
    stbi_image_free(decoded);
    view = {destination.data(), width, height, engine::PixelFormat::U8};
}

} // namespace

void prepare_vision(const Inputs & inputs, const ArtifactContract & artifact,
                    PreparedInputs & prepared) {
    const NamedTensorView * latent = nullptr;
    for (const NamedTensorView & value : inputs.history) {
        if (value.name != kGwp05ReferenceLatent) {
            throw Error(ErrorCode::unsupported, "unsupported GWP history tensor",
                        {{"history.name", value.name}});
        }
        if (latent != nullptr) invalid("GWP reference latent appears more than once",
                                      kGwp05ReferenceLatent, "duplicate");
        latent = &value;
    }
    if (latent != nullptr) {
        if (!inputs.images.empty()) {
            invalid("raw images and precomputed reference latent are mutually exclusive",
                    "images", "both supplied");
        }
        prepare_latent(*latent, artifact, prepared);
        return;
    }

    std::vector<std::string> names;
    names.reserve(inputs.images.size());
    for (const wam::ImageView & image : inputs.images) names.push_back(image.name);
    const std::array<std::size_t, 3> order = semantics::resolve_camera_order(names);
    prepared.images.resize(order.size());
    prepared.image_storage.resize(order.size());
    for (std::size_t canonical = 0; canonical < order.size(); ++canonical) {
        decode_image(inputs.images[order[canonical]], prepared.image_storage[canonical],
                     prepared.images[canonical]);
    }
    prepared.view.images = prepared.images.data();
    prepared.view.n_images = static_cast<int>(prepared.images.size());
}

} // namespace wam::internal::gwp05

namespace wam::internal::gwp05 {
namespace {

std::vector<std::int32_t> read_embedding_mask(const TensorView & tensor,
                                               std::size_t tokens) {
    if (tensor.empty()) return {};
    if (tensor.dtype != DType::i32 || tensor.byte_order != ByteOrder::little ||
        tensor.shape != std::vector<std::int64_t>{static_cast<std::int64_t>(tokens)} ||
        tensor.byte_size != tokens * sizeof(std::int32_t)) {
        invalid("embedding attention mask has the wrong tensor contract", "language.attention_mask",
                "expected little-endian I32 [T]");
    }
    const auto * values = static_cast<const std::int32_t *>(tensor.data);
    return {values, values + tokens};
}

std::vector<float> read_embedding(const TensorView & tensor,
                                  std::size_t & tokens, std::size_t hidden) {
    if (tensor.data == nullptr || tensor.shape.size() != 2 ||
        tensor.shape[0] <= 0 || tensor.shape[1] != static_cast<std::int64_t>(hidden) ||
        tensor.layout != "T,D" || tensor.byte_order != ByteOrder::little) {
        invalid("precomputed prompt embedding has the wrong tensor contract",
                "language.embedding", "expected little-endian [T,D] with layout T,D");
    }
    tokens = static_cast<std::size_t>(tensor.shape[0]);
    const std::size_t elements = tokens * hidden;
    std::vector<float> result(elements);
    if (tensor.dtype == DType::f32 && tensor.byte_size == elements * sizeof(float)) {
        std::memcpy(result.data(), tensor.data, tensor.byte_size);
    } else if (tensor.dtype == DType::bf16 &&
               tensor.byte_size == elements * sizeof(ggml_bf16_t)) {
        std::vector<ggml_bf16_t> values(elements);
        std::memcpy(values.data(), tensor.data, tensor.byte_size);
        ggml_bf16_to_fp32_row(values.data(), result.data(), static_cast<std::int64_t>(elements));
    } else {
        invalid("precomputed prompt embedding dtype or byte size is invalid",
                "language.embedding", "expected F32 or BF16 payload");
    }
    return result;
}

} // namespace

void prepare_language(const Inputs & inputs, const ArtifactContract & artifact,
                      PreparedInputs & prepared) {
    if (const auto * token_input = std::get_if<TokenInput>(&inputs.language)) {
        if (token_input->token_ids.empty() || token_input->attention_mask.empty() ||
            token_input->token_ids.size != token_input->attention_mask.size) {
            invalid("token IDs and attention mask must be nonempty and equal length",
                    "language.tokens", "shape mismatch");
        }
        prepared.tokens.assign(token_input->token_ids.data,
                               token_input->token_ids.data + token_input->token_ids.size);
        prepared.attention_mask.assign(
            token_input->attention_mask.data,
            token_input->attention_mask.data + token_input->attention_mask.size);
        (void) semantics::prepare_prompt(prepared.tokens, prepared.attention_mask,
                                         artifact.geometry.t5_vocab_size);
        prepared.view.lang_tokens = prepared.tokens.data();
        prepared.view.n_lang = static_cast<int>(prepared.tokens.size());
        prepared.view.attention_mask = prepared.attention_mask.data();
        prepared.view.attention_mask_n = static_cast<int>(prepared.attention_mask.size());
        return;
    }
    if (const auto * embedding_input = std::get_if<EmbeddingInput>(&inputs.language)) {
        std::size_t tokens = 0;
        std::vector<float> embedding = read_embedding(
            embedding_input->embedding, tokens, artifact.geometry.t5_hidden);
        const std::vector<std::int32_t> mask =
            read_embedding_mask(embedding_input->attention_mask, tokens);
        const semantics::PromptPlan plan = semantics::prepare_prompt(
            std::vector<std::int32_t>(tokens, 0), mask, 1);
        const std::size_t active_elements = plan.valid_tokens * artifact.geometry.t5_hidden;
        embedding.resize(active_elements);
        prepared.prompt_embedding = semantics::pad_prompt_embedding(
            embedding, plan.valid_tokens, artifact.geometry.t5_hidden);
        prepared.view.precomputed_prompt_emb = prepared.prompt_embedding.data();
        prepared.view.precomputed_prompt_tokens = static_cast<int>(semantics::kPromptTokens);
        return;
    }
    invalid("GWP language input is required", "language", "oneof is unset");
}

} // namespace wam::internal::gwp05
