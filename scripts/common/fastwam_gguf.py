"""FastWAM artifact constants and profile validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.policy_spec import POLICY_PROFILE_FORMAT, POLICY_SCHEMA_VERSION


ARCH = "fastwam"
CONVERSION_POLICY = "fastwam-bf16-policy-v2"
CONVERTER_REVISION = "wam-0.5-fastwam-policy-spec-v2"
EXPECTED_COMPONENT_COUNTS = {
    "video": 825,
    "action": 824,
    "vae": 86,
    "proprio": 2,
    "stats": 4,
}


def load_policy_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read PolicySpec profile {path}: {exc}") from exc
    if profile.get("format") != POLICY_PROFILE_FORMAT:
        raise ValueError("unsupported PolicySpec profile format")
    if profile.get("artifact_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("FastWAM PolicySpec schema version must be 2")
    for section in ("identity", "images", "state", "language", "action", "normalization"):
        if not isinstance(profile.get(section), dict):
            raise ValueError(f"PolicySpec section {section!r} is required")

    identity = profile["identity"]
    for field in ("profile", "checkpoint_revision", "training_dataset", "embodiment"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"PolicySpec identity.{field} must be nonempty")

    images = profile["images"]
    roles = images.get("roles")
    views = images.get("views")
    composition = images.get("composition")
    if not isinstance(roles, list) or len(roles) not in (2, 3) or len(set(roles)) != len(roles):
        raise ValueError("FastWAM requires two or three unique image roles")
    if not isinstance(views, dict) or set(views) != set(roles):
        raise ValueError("PolicySpec views must match image roles")
    if not isinstance(composition, dict) or composition.get("kind") != "canvas":
        raise ValueError("FastWAM requires canvas image composition")
    canvas_height = composition.get("height")
    canvas_width = composition.get("width")
    if not isinstance(canvas_height, int) or not isinstance(canvas_width, int) or canvas_height <= 0 or canvas_width <= 0:
        raise ValueError("FastWAM canvas dimensions must be positive")
    area = 0
    for role in roles:
        view = views[role]
        rect = view.get("rect") if isinstance(view, dict) else None
        if not isinstance(rect, list) or len(rect) != 4 or any(not isinstance(value, int) for value in rect):
            raise ValueError(f"invalid image placement for {role!r}")
        x, y, width, height = rect
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > canvas_width or y + height > canvas_height:
            raise ValueError(f"image placement for {role!r} is outside the canvas")
        if (view.get("target_height"), view.get("target_width")) != (height, width):
            raise ValueError(f"image target for {role!r} differs from its placement")
        if view.get("resize_mode") not in ("none", "stretch", "cover_center_crop"):
            raise ValueError(f"invalid resize mode for {role!r}")
        if view.get("interpolation") not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(f"invalid interpolation for {role!r}")
        if not isinstance(view.get("antialias"), bool):
            raise ValueError(f"antialias for {role!r} must be boolean")
        area += width * height
    if area != canvas_height * canvas_width:
        raise ValueError("FastWAM image placements must cover the canvas")
    if images.get("color_space") != "rgb" or images.get("pixel_range") != "minus_one_to_one" or images.get("tensor_layout") != "chw":
        raise ValueError("FastWAM image output must be RGB CHW in [-1,1]")

    state = profile["state"]
    action = profile["action"]
    for name, value in (("state", state), ("action", action)):
        real_dim = value.get("real_dim")
        model_dim = value.get("model_dim")
        fields = value.get("fields")
        if not isinstance(real_dim, int) or real_dim <= 0 or model_dim != real_dim:
            raise ValueError(f"FastWAM {name} dimensions must be positive and unpadded")
        if not isinstance(fields, list) or len(fields) != real_dim or len(set(fields)) != len(fields):
            raise ValueError(f"FastWAM {name} fields must match its dimension")
    if not isinstance(action.get("horizon"), int) or action["horizon"] <= 0:
        raise ValueError("FastWAM action horizon must be positive")
    if action.get("recovery") != {"kind": "identity"}:
        raise ValueError("FastWAM Gate B action recovery must be identity")

    language = profile["language"]
    if language.get("input_mode") != "embedding" or language.get("text_encoder_in_artifact") is not False:
        raise ValueError("FastWAM Gate B requires external embedding input")
    if not isinstance(language.get("max_tokens"), int) or language["max_tokens"] <= 0:
        raise ValueError("FastWAM language max_tokens must be positive")
    if language.get("attention_mask_required") is not True:
        raise ValueError("FastWAM external embedding requires an attention mask")

    normalization = profile["normalization"]
    for domain in ("state", "action"):
        spec = normalization.get(domain)
        if not isinstance(spec, dict) or spec.get("kind") != "min_max" or spec.get("clip") is not False:
            raise ValueError(f"FastWAM LIBERO {domain} normalization must be unclipped min_max")
    if normalization["state"].get("output_clamp") != [-5.0, 5.0]:
        raise ValueError("FastWAM LIBERO state output clamp must be [-5,5]")
    if "output_clamp" in normalization["action"]:
        raise ValueError("FastWAM LIBERO action normalization must not clamp")
    epsilon = normalization.get("epsilon")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")
    return profile


def publish_no_overwrite(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}") from exc
    temporary.unlink()
