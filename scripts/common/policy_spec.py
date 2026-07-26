"""Architecture-neutral PolicySpec GGUF metadata serialization."""

from __future__ import annotations

from typing import Any, Mapping

from gguf import GGUFValueType


POLICY_PROFILE_FORMAT = "wam-policy-spec-profile-v1"
POLICY_SCHEMA_VERSION = 2


def add_policy_spec_metadata(writer: Any, profile: Mapping[str, Any]) -> None:
    identity = profile["identity"]
    images = profile["images"]
    state = profile["state"]
    language = profile["language"]
    action = profile["action"]
    normalization = profile["normalization"]

    writer.add_uint32("wam.artifact_schema_version", profile["artifact_schema_version"])
    for field in ("profile", "checkpoint_revision", "training_dataset", "embodiment"):
        writer.add_string(f"wam.policy.{field}", identity[field])
    writer.add_array("wam.input.image.roles", images["roles"])
    for role in images["roles"]:
        view = images["views"][role]
        prefix = f"wam.input.image.{role}."
        writer.add_uint32(prefix + "target_height", view["target_height"])
        writer.add_uint32(prefix + "target_width", view["target_width"])
        writer.add_string(prefix + "resize_mode", view["resize_mode"])
        writer.add_string(prefix + "interpolation", view["interpolation"])
        writer.add_bool(prefix + "antialias", view["antialias"])
    composition = images["composition"]
    writer.add_string("wam.input.image.composition.kind", composition["kind"])
    writer.add_uint32("wam.input.image.composition.height", composition["height"])
    writer.add_uint32("wam.input.image.composition.width", composition["width"])
    for role in images["roles"]:
        writer.add_key_value(
            f"wam.input.image.composition.{role}.rect",
            images["views"][role]["rect"], GGUFValueType.ARRAY,
            GGUFValueType.UINT32)
    writer.add_string("wam.input.image.color_space", images["color_space"])
    writer.add_string("wam.input.image.pixel_range", images["pixel_range"])
    writer.add_string("wam.input.image.tensor_layout", images["tensor_layout"])

    writer.add_uint32("wam.input.state.real_dim", state["real_dim"])
    writer.add_uint32("wam.input.state.model_dim", state["model_dim"])
    writer.add_float32("wam.input.state.pad_value", state["pad_value"])
    writer.add_array("wam.input.state.fields", state["fields"])
    for field in ("input_mode", "prompt_template", "tokenizer_family", "tokenizer_revision"):
        writer.add_string(f"wam.input.language.{field}", language[field])
    writer.add_uint32("wam.input.language.max_tokens", language["max_tokens"])
    writer.add_bool("wam.input.language.text_encoder_in_artifact", language["text_encoder_in_artifact"])
    for field in ("padding_side", "truncation_side"):
        writer.add_string(f"wam.input.language.{field}", language[field])
    writer.add_bool("wam.input.language.attention_mask_required", language["attention_mask_required"])
    special_token_ids = language["special_token_ids"]
    if special_token_ids:
        writer.add_key_value(
            "wam.input.language.special_token_ids", special_token_ids,
            GGUFValueType.ARRAY, GGUFValueType.INT32)

    writer.add_uint32("wam.output.action.horizon", action["horizon"])
    writer.add_uint32("wam.output.action.real_dim", action["real_dim"])
    writer.add_uint32("wam.output.action.model_dim", action["model_dim"])
    writer.add_array("wam.output.action.fields", action["fields"])
    for field in ("representation", "frame", "gripper"):
        writer.add_string(f"wam.output.action.{field}", action[field])
    writer.add_string("wam.output.action.recovery.kind", action["recovery"]["kind"])
    indices = action["recovery"].get("reference_state_indices")
    if indices:
        writer.add_key_value(
            "wam.output.action.recovery.reference_state_indices", indices,
            GGUFValueType.ARRAY, GGUFValueType.INT32)

    for domain in ("state", "action"):
        writer.add_string(
            f"wam.normalization.{domain}.kind", normalization[domain]["kind"])
        writer.add_bool(
            f"wam.normalization.{domain}.clip", normalization[domain]["clip"])
        clamp = normalization[domain].get("output_clamp")
        if clamp is not None:
            writer.add_float32(
                f"wam.normalization.{domain}.output_clamp_lower", clamp[0])
            writer.add_float32(
                f"wam.normalization.{domain}.output_clamp_upper", clamp[1])
    writer.add_float32("wam.normalization.epsilon", normalization["epsilon"])
