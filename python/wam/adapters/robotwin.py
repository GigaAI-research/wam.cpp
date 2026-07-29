from __future__ import annotations

import numpy as np

from .base import Adapter, EnvironmentContract

JOINT_FIELDS = tuple(
    f"{arm}.joint.{index}" if index < 6 else f"{arm}.gripper"
    for arm in ("left", "right") for index in range(7))
ROLE_MAP = {
    "observation.images.cam_high": "camera_high",
    "observation.images.cam_left_wrist": "camera_left_wrist",
    "observation.images.cam_right_wrist": "camera_right_wrist",
}
CONTRACT = EnvironmentContract(
    "robotwin", tuple(ROLE_MAP.values()), JOINT_FIELDS, JOINT_FIELDS,
    14, 14, "joint_position", "controller", "continuous")


def rgb_u8(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"RoboTwin image must be HWC RGB/RGBA, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255)
    return np.ascontiguousarray(array, dtype=np.uint8)


class RoboTwinAdapter(Adapter):
    contract = CONTRACT

    def validate(self, spec):
        roles = {view.role for view in spec.images.views}
        if roles != set(ROLE_MAP.values()):
            raise ValueError(f"RoboTwin image roles differ: {sorted(roles)}")
        if int(spec.state.real_dim) != 14 or int(spec.action.real_dim) != 14:
            raise ValueError("RoboTwin requires 14D state and action")
        if tuple(spec.state.fields) != JOINT_FIELDS or \
                tuple(spec.action.fields) != JOINT_FIELDS:
            raise ValueError("RoboTwin joint field order differs from PolicySpec")
        if spec.action.representation != "joint_position" or \
                spec.action.frame != "controller":
            raise ValueError(
                "RoboTwin requires controller-frame joint_position actions")

    def observation(self, value, spec):
        self.validate(spec)
        images = [{"name": target, "data": rgb_u8(value[source])}
                  for source, target in ROLE_MAP.items()]
        state = np.ascontiguousarray(
            value["observation.state"], dtype=np.float32).reshape(-1)
        if state.shape != (14,):
            raise ValueError(f"RoboTwin state must be 14D, got {state.shape}")
        return images, state

    def action(self, value, spec):
        self.validate(spec)
        action = np.ascontiguousarray(value, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 14:
            raise ValueError(f"RoboTwin action chunk must be [T,14], got {action.shape}")
        return action
