"""Shared GWP-0.5 GGUF policy and artifact I/O helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from gguf import GGUFValueType


ARCH = "gwp05"
CONVERTER_REVISION = "wam-0.5-gwp05-policy-spec-v2"
SUPPORTED_CONVERTER_REVISIONS = {
    "gwp05-native-bf16-v1",
    CONVERTER_REVISION,
}
POLICY_PROFILE_FORMAT = "wam-policy-spec-profile-v1"
POLICY_SCHEMA_VERSION = 2
DTYPE_BYTES = {"F32": 4, "BF16": 2}
STATISTIC_NAMES = (
    "state_mean", "state_std", "state_q01", "state_q99",
    "action_mean", "action_std", "action_q01", "action_q99",
)
POLICY_STATISTIC_NAMES = (
    "wam.norm.state.mean", "wam.norm.state.std",
    "wam.norm.state.q01", "wam.norm.state.q99",
    "wam.norm.action.mean", "wam.norm.action.std",
    "wam.norm.action.q01", "wam.norm.action.q99",
)
POLICIES: dict[str, dict[str, Any]] = {
    "source-f32": {
        "id": "source-f32-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
    },
    "mot-bf16": {
        "id": "mot-bf16-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "F32", "stats": "F32"},
    },
    "mot-vae-bf16": {
        "id": "mot-vae-bf16-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "BF16", "stats": "F32"},
    },
    "mot-vae-bf16-qkv": {
        "id": "mot-vae-bf16-qkv-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "BF16", "stats": "F32"},
        "pack_self_qkv": True,
    },
}
POLICIES_BY_ID = {value["id"]: value for value in POLICIES.values()}


def kv(name: str) -> str:
    return f"{ARCH}.{name}"


def policy(name: str) -> dict[str, Any]:
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown conversion policy {name!r}") from exc


def component_expected_counts(layers: int, t5_layers: int, packed: bool) -> dict[str, int]:
    return {
        "gwp": 44 + (46 if packed else 54) * layers,
        "t5": 2 + 10 * t5_layers,
        "vae": 82,
        "stats": 8,
    }


def load_policy_profile(path: Path, *, model_dim: int, action_horizon: int,
                        image_height: int, image_width: int) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PolicySpec profile {path}: {exc}") from exc
    if profile.get("format") != POLICY_PROFILE_FORMAT:
        raise ValueError(f"unsupported PolicySpec profile format: {profile.get('format')!r}")
    if profile.get("artifact_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("PolicySpec profile schema version must be 2")

    for section in ("identity", "images", "state", "language", "action", "normalization"):
        if not isinstance(profile.get(section), dict):
            raise ValueError(f"PolicySpec profile section {section!r} is required")
    identity = profile["identity"]
    for field in ("profile", "checkpoint_revision", "training_dataset", "embodiment"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"PolicySpec identity.{field} must be a nonempty string")

    images = profile["images"]
    roles = images.get("roles")
    views = images.get("views")
    composition = images.get("composition")
    if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
        raise ValueError("PolicySpec image roles must be a nonempty unique list")
    if not isinstance(views, dict) or set(views) != set(roles):
        raise ValueError("PolicySpec image views must match roles exactly")
    if not isinstance(composition, dict) or composition.get("kind") != "canvas":
        raise ValueError("GWP05 PolicySpec requires canvas image composition")
    if (composition.get("height"), composition.get("width")) != (
            image_height, image_width):
        raise ValueError("PolicySpec canvas dimensions differ from converter arguments")
    covered_area = 0
    for role in roles:
        view = views[role]
        if not isinstance(view, dict):
            raise ValueError(f"PolicySpec image view {role!r} must be an object")
        rect = view.get("rect")
        if not isinstance(rect, list) or len(rect) != 4 or any(
                not isinstance(value, int) or value < 0 for value in rect):
            raise ValueError(f"PolicySpec image rect for {role!r} is invalid")
        x, y, width, height = rect
        if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
            raise ValueError(f"PolicySpec image rect for {role!r} is outside the canvas")
        if (view.get("target_height"), view.get("target_width")) != (height, width):
            raise ValueError(f"PolicySpec image target for {role!r} differs from its rect")
        if view.get("resize_mode") not in ("none", "stretch", "cover_center_crop"):
            raise ValueError(f"PolicySpec image resize mode for {role!r} is invalid")
        if view.get("interpolation") not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(f"PolicySpec image interpolation for {role!r} is invalid")
        if not isinstance(view.get("antialias"), bool):
            raise ValueError(f"PolicySpec image antialias for {role!r} must be boolean")
        covered_area += width * height
    if covered_area != image_height * image_width:
        raise ValueError("PolicySpec image rects do not cover the canvas")

    state = profile["state"]
    action = profile["action"]
    if state.get("model_dim") != model_dim or action.get("model_dim") != model_dim:
        raise ValueError("PolicySpec model dimensions differ from the checkpoint")
    if action.get("horizon") != action_horizon:
        raise ValueError("PolicySpec action horizon differs from the converter argument")
    for name, value in (("state", state), ("action", action)):
        real_dim = value.get("real_dim")
        fields = value.get("fields")
        if not isinstance(real_dim, int) or real_dim <= 0 or real_dim > model_dim:
            raise ValueError(f"PolicySpec {name} real dimension is invalid")
        if not isinstance(fields, list) or len(fields) != real_dim or len(set(fields)) != len(fields):
            raise ValueError(f"PolicySpec {name} fields do not match real_dim")

    language = profile["language"]
    required_language = {
        "input_mode", "prompt_template", "tokenizer_family", "tokenizer_revision",
        "max_tokens", "text_encoder_in_artifact", "padding_side", "truncation_side",
        "attention_mask_required", "special_token_ids",
    }
    if not required_language.issubset(language):
        raise ValueError("PolicySpec language contract is incomplete")
    if language["input_mode"] not in ("tokens", "embedding", "tokens_or_embedding"):
        raise ValueError("PolicySpec language input mode is invalid")

    normalization = profile["normalization"]
    for domain in ("state", "action"):
        value = normalization.get(domain)
        if not isinstance(value, dict) or value.get("kind") != "z_score" or value.get("clip") is not False:
            raise ValueError(f"RoboTwin PolicySpec {domain} normalization must be unclipped z_score")
    epsilon = normalization.get("epsilon")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("PolicySpec normalization epsilon must be positive")

    recovery = action.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("kind") != "add_current_state":
        raise ValueError("RoboTwin PolicySpec requires add_current_state recovery")
    indices = recovery.get("reference_state_indices")
    if not isinstance(indices, list) or len(indices) != action["real_dim"] or any(
            not isinstance(index, int) or index < -1 or index >= state["real_dim"]
            for index in indices):
        raise ValueError("PolicySpec action recovery indices are invalid")
    return profile


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
    writer.add_key_value(
        "wam.input.language.special_token_ids", language["special_token_ids"],
        GGUFValueType.ARRAY, GGUFValueType.INT32)

    writer.add_uint32("wam.output.action.horizon", action["horizon"])
    writer.add_uint32("wam.output.action.real_dim", action["real_dim"])
    writer.add_uint32("wam.output.action.model_dim", action["model_dim"])
    writer.add_array("wam.output.action.fields", action["fields"])
    for field in ("representation", "frame", "gripper"):
        writer.add_string(f"wam.output.action.{field}", action[field])
    writer.add_string("wam.output.action.recovery.kind", action["recovery"]["kind"])
    writer.add_key_value(
        "wam.output.action.recovery.reference_state_indices",
        action["recovery"]["reference_state_indices"],
        GGUFValueType.ARRAY, GGUFValueType.INT32)

    for domain in ("state", "action"):
        writer.add_string(
            f"wam.normalization.{domain}.kind", normalization[domain]["kind"])
        writer.add_bool(
            f"wam.normalization.{domain}.clip", normalization[domain]["clip"])
    writer.add_float32("wam.normalization.epsilon", normalization["epsilon"])


def add_policy_metadata(writer: Any, definition: Mapping[str, Any],
                        component_stats: Mapping[str, Mapping[str, Any]] | None) -> None:
    """Write the explicit policy and component telemetry shared with inspection."""
    writer.add_string(kv("weight_policy"), definition["id"])
    writer.add_string(kv("conversion_policy"), definition["id"])
    writer.add_string(kv("converter_revision"), CONVERTER_REVISION)
    if component_stats is None:
        return
    for component in ("gwp", "t5", "vae", "stats"):
        values = component_stats[component]
        writer.add_string(kv(f"{component}_source_dtype"), str(values["source_dtype"]))
        writer.add_string(kv(f"{component}_stored_dtype"), str(values["stored_dtype"]))
        for field in ("tensor_count", "elements", "source_bytes", "stored_bytes"):
            writer.add_uint64(kv(f"{component}_{field}"), int(values[field]))


def publish_no_overwrite(temporary: Path, destination: Path) -> None:
    """Atomically publish a completed file while refusing replacement."""
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}") from exc
    temporary.unlink()


def write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
