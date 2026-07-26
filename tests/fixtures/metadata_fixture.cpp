#include "fixtures/metadata_fixture.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace wam::test {
namespace {

using MetadataValue = std::variant<
    std::string,
    std::uint32_t,
    float,
    bool,
    std::vector<std::string>,
    std::vector<std::uint32_t>,
    std::vector<std::int32_t>>;

void set_metadata(gguf_context * context, const std::string & key,
                  const MetadataValue & value) {
    std::visit(
        [&](const auto & item) {
            using Value = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<Value, std::string>) {
                gguf_set_val_str(context, key.c_str(), item.c_str());
            } else if constexpr (std::is_same_v<Value, std::uint32_t>) {
                gguf_set_val_u32(context, key.c_str(), item);
            } else if constexpr (std::is_same_v<Value, float>) {
                gguf_set_val_f32(context, key.c_str(), item);
            } else if constexpr (std::is_same_v<Value, bool>) {
                gguf_set_val_bool(context, key.c_str(), item);
            } else if constexpr (
                std::is_same_v<Value, std::vector<std::string>>) {
                std::vector<const char *> strings;
                strings.reserve(item.size());
                for (const std::string & string : item) {
                    strings.push_back(string.c_str());
                }
                gguf_set_arr_str(context, key.c_str(), strings.data(),
                                 strings.size());
            } else if constexpr (
                std::is_same_v<Value, std::vector<std::uint32_t>>) {
                gguf_set_arr_data(context, key.c_str(), GGUF_TYPE_UINT32,
                                  item.data(), item.size());
            } else if constexpr (
                std::is_same_v<Value, std::vector<std::int32_t>>) {
                gguf_set_arr_data(context, key.c_str(), GGUF_TYPE_INT32,
                                  item.data(), item.size());
            }
        },
        value);
}

std::vector<std::string> dual_arm_fields() {
    return {
        "left.joint.0", "left.joint.1", "left.joint.2",
        "left.joint.3", "left.joint.4", "left.joint.5",
        "left.gripper", "right.joint.0", "right.joint.1",
        "right.joint.2", "right.joint.3", "right.joint.4",
        "right.joint.5", "right.gripper",
    };
}

void add_gwp05_geometry(MetadataFixture & fixture) {
    fixture.set_string("gwp05.architecture", "gwp05");
    fixture.set_string("gwp05.conversion_policy", "source-f32-v1");
    fixture.set_string("gwp05.weight_policy", "source-f32-v1");
    for (const auto & item :
         std::vector<std::pair<const char *, std::uint32_t>>{
             {"hidden", 128}, {"n_layers", 2}, {"n_heads", 1},
             {"head_dim", 128}, {"ffn_dim", 256},
             {"action_hidden", 64}, {"action_ffn_dim", 128},
             {"action_dim", 32}, {"real_state_dim", 14},
             {"real_action_dim", 14}, {"num_embodiments", 1},
             {"embodiment_id", 0}, {"image_height", 32},
             {"image_width", 32}, {"num_views", 3},
             {"action_chunk", 48}, {"inference_steps", 10},
             {"t5_vocab_size", 128}, {"t5_hidden", 64},
             {"t5_ffn_dim", 128}, {"t5_heads", 1},
             {"t5_head_dim", 64}, {"t5_layers", 2},
             {"t5_max_length", 512}, {"vae_z_dim", 16},
         }) {
        fixture.set_u32(std::string("gwp05.") + item.first, item.second);
    }
    fixture.set_f32("gwp05.flow_shift", 5.0F);
    fixture.set_f32("gwp05.norm_eps", 1.0e-6F);
}

void add_gwp05_statistics(MetadataFixture & fixture,
                          const std::string & prefix) {
    std::vector<float> lower(32, -1.0F);
    std::vector<float> upper(32, 1.0F);
    std::vector<float> mask(32, 0.0F);
    std::fill_n(mask.begin(), 14, 1.0F);
    fixture.set_f32_tensor(prefix + "state.q01", lower);
    fixture.set_f32_tensor(prefix + "state.q99", upper);
    fixture.set_f32_tensor(prefix + "state.mask", mask);
    fixture.set_f32_tensor(prefix + "action.q01", std::move(lower));
    fixture.set_f32_tensor(prefix + "action.q99", std::move(upper));
    fixture.set_f32_tensor(prefix + "action.mask", std::move(mask));
}

} // namespace

struct MetadataFixture::Impl {
    std::map<std::string, MetadataValue> metadata;
    std::map<std::string, std::vector<float>> tensors;
};

MetadataFixture::MetadataFixture() : impl_(std::make_unique<Impl>()) {}

MetadataFixture::~MetadataFixture() = default;
MetadataFixture::MetadataFixture(MetadataFixture &&) noexcept = default;
MetadataFixture & MetadataFixture::operator=(MetadataFixture &&) noexcept =
    default;

void MetadataFixture::set_string(std::string key, std::string value) {
    impl_->metadata[std::move(key)] = std::move(value);
}

void MetadataFixture::set_u32(std::string key, std::uint32_t value) {
    impl_->metadata[std::move(key)] = value;
}

void MetadataFixture::set_f32(std::string key, float value) {
    impl_->metadata[std::move(key)] = value;
}

void MetadataFixture::set_bool(std::string key, bool value) {
    impl_->metadata[std::move(key)] = value;
}

void MetadataFixture::set_string_array(std::string key,
                                       std::vector<std::string> value) {
    impl_->metadata[std::move(key)] = std::move(value);
}

void MetadataFixture::set_u32_array(std::string key,
                                    std::vector<std::uint32_t> value) {
    impl_->metadata[std::move(key)] = std::move(value);
}

void MetadataFixture::set_i32_array(std::string key,
                                    std::vector<std::int32_t> value) {
    impl_->metadata[std::move(key)] = std::move(value);
}

void MetadataFixture::set_f32_tensor(std::string name,
                                     std::vector<float> value) {
    if (value.empty()) {
        throw std::invalid_argument("fixture tensors must not be empty");
    }
    impl_->tensors[std::move(name)] = std::move(value);
}

void MetadataFixture::remove(const std::string & key) {
    impl_->metadata.erase(key);
    impl_->tensors.erase(key);
}

void MetadataFixture::write(const std::string & path) const {
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        throw std::runtime_error("cannot create GGUF fixture context");
    }

    std::size_t tensor_bytes = 0;
    for (const auto & tensor : impl_->tensors) {
        tensor_bytes += tensor.second.size() * sizeof(float);
    }
    ggml_init_params parameters{};
    parameters.mem_size = tensor_bytes +
        (impl_->tensors.size() + 8) * ggml_tensor_overhead() + 4096;
    parameters.no_alloc = false;
    ggml_context * tensors = ggml_init(parameters);
    if (tensors == nullptr) {
        gguf_free(gguf);
        throw std::runtime_error("cannot create GGML fixture context");
    }

    try {
        for (const auto & entry : impl_->metadata) {
            set_metadata(gguf, entry.first, entry.second);
        }
        for (const auto & entry : impl_->tensors) {
            ggml_tensor * tensor = ggml_new_tensor_1d(
                tensors, GGML_TYPE_F32,
                static_cast<std::int64_t>(entry.second.size()));
            ggml_set_name(tensor, entry.first.c_str());
            std::memcpy(tensor->data, entry.second.data(),
                        entry.second.size() * sizeof(float));
            gguf_add_tensor(gguf, tensor);
        }
        if (!gguf_write_to_file(gguf, path.c_str(), false)) {
            throw std::runtime_error("cannot write GGUF metadata fixture");
        }
    } catch (...) {
        gguf_free(gguf);
        ggml_free(tensors);
        throw;
    }
    gguf_free(gguf);
    ggml_free(tensors);
}

MetadataFixture valid_policy_fixture() {
    MetadataFixture fixture;
    fixture.set_string("general.architecture", "gwp05");
    fixture.set_u32("wam.artifact_schema_version", 2);
    fixture.set_string("wam.policy.profile", "synthetic_2cam_joint");
    fixture.set_string("wam.policy.embodiment", "synthetic_arm");
    fixture.set_string("wam.policy.training_dataset", "synthetic");
    fixture.set_string("wam.policy.checkpoint_revision", "fixture-v1");

    fixture.set_string_array("wam.input.image.roles", {"scene", "wrist"});
    for (const std::string role : {"scene", "wrist"}) {
        const std::string prefix = "wam.input.image." + role + ".";
        fixture.set_u32(prefix + "target_height", 2);
        fixture.set_u32(prefix + "target_width", 2);
        fixture.set_string(prefix + "resize_mode", "stretch");
        fixture.set_string(prefix + "interpolation", "bilinear");
        fixture.set_bool(prefix + "antialias", true);
    }
    fixture.set_string("wam.input.image.composition.kind", "canvas");
    fixture.set_u32("wam.input.image.composition.height", 2);
    fixture.set_u32("wam.input.image.composition.width", 4);
    fixture.set_u32_array("wam.input.image.composition.scene.rect",
                          {0, 0, 2, 2});
    fixture.set_u32_array("wam.input.image.composition.wrist.rect",
                          {2, 0, 2, 2});
    fixture.set_string("wam.input.image.color_space", "rgb");
    fixture.set_string("wam.input.image.pixel_range", "minus_one_to_one");
    fixture.set_string("wam.input.image.tensor_layout", "chw");

    fixture.set_u32("wam.input.state.real_dim", 3);
    fixture.set_u32("wam.input.state.model_dim", 4);
    fixture.set_f32("wam.input.state.pad_value", 0.0F);
    fixture.set_string_array("wam.input.state.fields",
                             {"joint.0", "joint.1", "gripper"});

    fixture.set_string("wam.input.language.input_mode", "tokens");
    fixture.set_string("wam.input.language.prompt_template", "Act: {task}");
    fixture.set_string("wam.input.language.tokenizer_family", "synthetic");
    fixture.set_string("wam.input.language.tokenizer_revision", "fixture-v1");
    fixture.set_u32("wam.input.language.max_tokens", 16);
    fixture.set_bool("wam.input.language.text_encoder_in_artifact", true);
    fixture.set_string("wam.input.language.padding_side", "right");
    fixture.set_string("wam.input.language.truncation_side", "right");
    fixture.set_bool("wam.input.language.attention_mask_required", true);
    fixture.set_i32_array("wam.input.language.special_token_ids", {0, 1});

    fixture.set_u32("wam.output.action.horizon", 2);
    fixture.set_u32("wam.output.action.real_dim", 3);
    fixture.set_u32("wam.output.action.model_dim", 4);
    fixture.set_string_array("wam.output.action.fields",
                             {"joint.0", "joint.1", "gripper"});
    fixture.set_string("wam.output.action.representation", "joint_position");
    fixture.set_string("wam.output.action.frame", "controller");
    fixture.set_string("wam.output.action.gripper", "continuous");
    fixture.set_string("wam.output.action.recovery.kind",
                       "add_current_state");
    fixture.set_i32_array(
        "wam.output.action.recovery.reference_state_indices", {0, 1, -1});

    fixture.set_string("wam.normalization.state.kind", "min_max");
    fixture.set_bool("wam.normalization.state.clip", true);
    fixture.set_string("wam.normalization.action.kind", "min_max");
    fixture.set_bool("wam.normalization.action.clip", true);
    fixture.set_f32("wam.normalization.epsilon", 1.0e-6F);
    fixture.set_f32_tensor("wam.norm.state.q01", {-1, -2, -3, 0});
    fixture.set_f32_tensor("wam.norm.state.q99", {1, 2, 3, 1});
    fixture.set_f32_tensor("wam.norm.action.q01", {-1, -2, -3, 0});
    fixture.set_f32_tensor("wam.norm.action.q99", {1, 2, 3, 1});
    return fixture;
}

MetadataFixture valid_gwp05_policy_fixture() {
    MetadataFixture fixture;
    fixture.set_string("general.architecture", "gwp05");
    add_gwp05_geometry(fixture);

    fixture.set_u32("wam.artifact_schema_version", 2);
    fixture.set_string("wam.policy.profile",
                       "gwp05_robotwin_dual_arm_fixture");
    fixture.set_string("wam.policy.embodiment", "dual_arm");
    fixture.set_string("wam.policy.training_dataset", "robotwin");
    fixture.set_string("wam.policy.checkpoint_revision", "fixture-v1");

    const std::vector<std::string> roles = {
        "scene", "left_wrist", "right_wrist"};
    fixture.set_string_array("wam.input.image.roles", roles);
    const std::vector<std::vector<std::uint32_t>> rectangles = {
        {0, 0, 32, 16}, {0, 16, 16, 16}, {16, 16, 16, 16}};
    for (std::size_t index = 0; index < roles.size(); ++index) {
        const std::string prefix = "wam.input.image." + roles[index] + ".";
        fixture.set_u32(prefix + "target_height", rectangles[index][3]);
        fixture.set_u32(prefix + "target_width", rectangles[index][2]);
        fixture.set_string(prefix + "resize_mode", "stretch");
        fixture.set_string(prefix + "interpolation", "bilinear");
        fixture.set_bool(prefix + "antialias", true);
        fixture.set_u32_array(
            "wam.input.image.composition." + roles[index] + ".rect",
            rectangles[index]);
    }
    fixture.set_string("wam.input.image.composition.kind", "canvas");
    fixture.set_u32("wam.input.image.composition.height", 32);
    fixture.set_u32("wam.input.image.composition.width", 32);
    fixture.set_string("wam.input.image.color_space", "rgb");
    fixture.set_string("wam.input.image.pixel_range", "minus_one_to_one");
    fixture.set_string("wam.input.image.tensor_layout", "chw");

    fixture.set_u32("wam.input.state.real_dim", 14);
    fixture.set_u32("wam.input.state.model_dim", 32);
    fixture.set_f32("wam.input.state.pad_value", 0.0F);
    fixture.set_string_array("wam.input.state.fields", dual_arm_fields());

    fixture.set_string("wam.input.language.input_mode",
                       "tokens_or_embedding");
    fixture.set_string("wam.input.language.prompt_template", "{task}");
    fixture.set_string("wam.input.language.tokenizer_family", "umt5");
    fixture.set_string("wam.input.language.tokenizer_revision", "fixture-v1");
    fixture.set_u32("wam.input.language.max_tokens", 64);
    fixture.set_bool("wam.input.language.text_encoder_in_artifact", true);
    fixture.set_string("wam.input.language.padding_side", "right");
    fixture.set_string("wam.input.language.truncation_side", "right");
    fixture.set_bool("wam.input.language.attention_mask_required", true);
    fixture.set_i32_array("wam.input.language.special_token_ids", {0, 1});

    fixture.set_u32("wam.output.action.horizon", 48);
    fixture.set_u32("wam.output.action.real_dim", 14);
    fixture.set_u32("wam.output.action.model_dim", 32);
    fixture.set_string_array("wam.output.action.fields", dual_arm_fields());
    fixture.set_string("wam.output.action.representation", "joint_position");
    fixture.set_string("wam.output.action.frame", "controller");
    fixture.set_string("wam.output.action.gripper", "continuous");
    fixture.set_string("wam.output.action.recovery.kind",
                       "add_current_state");
    fixture.set_i32_array(
        "wam.output.action.recovery.reference_state_indices",
        {0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, 11, 12, -1});

    fixture.set_string("wam.normalization.state.kind", "quantile");
    fixture.set_bool("wam.normalization.state.clip", false);
    fixture.set_string("wam.normalization.action.kind", "quantile");
    fixture.set_bool("wam.normalization.action.clip", false);
    fixture.set_f32("wam.normalization.epsilon", 1.0e-6F);
    add_gwp05_statistics(fixture, "wam.norm.");
    return fixture;
}

MetadataFixture valid_gwp05_robotwin_14d_policy_fixture() {
    MetadataFixture fixture = valid_gwp05_policy_fixture();
    fixture.set_string("wam.policy.profile",
                       "gwp05_robotwin_dual_arm_14d_zscore");
    fixture.set_string("wam.policy.training_dataset", "robotwin");
    fixture.set_string("wam.policy.checkpoint_revision",
                       "checkpoint_epoch_9_step_100000");

    fixture.set_u32("gwp05.action_dim", 14);
    fixture.set_u32("gwp05.num_embodiments", 1);
    fixture.set_u32("gwp05.image_height", 384);
    fixture.set_u32("gwp05.image_width", 320);
    fixture.set_u32("wam.input.state.model_dim", 14);
    fixture.set_u32("wam.output.action.model_dim", 14);

    const std::vector<std::string> roles = {
        "camera_high", "camera_left_wrist", "camera_right_wrist"};
    fixture.set_string_array("wam.input.image.roles", roles);
    const std::vector<std::vector<std::uint32_t>> rectangles = {
        {0, 0, 320, 256}, {0, 256, 160, 128}, {160, 256, 160, 128}};
    for (std::size_t index = 0; index < roles.size(); ++index) {
        const std::string prefix = "wam.input.image." + roles[index] + ".";
        fixture.set_u32(prefix + "target_height", rectangles[index][3]);
        fixture.set_u32(prefix + "target_width", rectangles[index][2]);
        fixture.set_string(prefix + "resize_mode", "cover_center_crop");
        fixture.set_string(prefix + "interpolation", "bilinear");
        fixture.set_bool(prefix + "antialias", true);
        fixture.set_u32_array(
            "wam.input.image.composition." + roles[index] + ".rect",
            rectangles[index]);
    }
    fixture.set_u32("wam.input.image.composition.height", 384);
    fixture.set_u32("wam.input.image.composition.width", 320);

    fixture.set_string("wam.normalization.state.kind", "z_score");
    fixture.set_string("wam.normalization.action.kind", "z_score");
    for (const char * domain : {"state", "action"}) {
        const std::string prefix = std::string("wam.norm.") + domain;
        fixture.remove(prefix + ".q01");
        fixture.remove(prefix + ".q99");
        fixture.remove(prefix + ".mask");
        fixture.set_f32_tensor(prefix + ".mean",
                               std::vector<float>(14, 0.0F));
        fixture.set_f32_tensor(prefix + ".std",
                               std::vector<float>(14, 1.0F));
    }
    return fixture;
}

MetadataFixture valid_gwp05_legacy_fixture() {
    MetadataFixture fixture;
    fixture.set_string("general.architecture", "gwp05");
    add_gwp05_geometry(fixture);
    fixture.remove("gwp05.conversion_policy");
    fixture.set_string("gwp05.weight_policy", "source");
    std::vector<float> lower(32, -1.0F);
    std::vector<float> upper(32, 1.0F);
    fixture.set_f32_tensor("state_q01", lower);
    fixture.set_f32_tensor("state_q99", upper);
    fixture.set_f32_tensor("action_q01", std::move(lower));
    fixture.set_f32_tensor("action_q99", std::move(upper));
    return fixture;
}

} // namespace wam::test
