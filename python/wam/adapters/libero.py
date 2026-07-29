from __future__ import annotations

import math
import numpy as np

from .base import Adapter, EnvironmentContract

STATE_FIELDS = (
    "eef.position.x", "eef.position.y", "eef.position.z",
    "eef.rotation.axis_angle.x", "eef.rotation.axis_angle.y",
    "eef.rotation.axis_angle.z", "gripper.left", "gripper.right")
ACTION_FIELDS = (
    "eef.delta.position.x", "eef.delta.position.y",
    "eef.delta.position.z", "eef.delta.rotation.axis_angle.x",
    "eef.delta.rotation.axis_angle.y",
    "eef.delta.rotation.axis_angle.z", "gripper.command")
CONTRACT = EnvironmentContract(
    "libero", ("scene", "wrist"), STATE_FIELDS, ACTION_FIELDS,
    8, 7, "eef_delta_pose", "robot_base", "continuous")


def quat_to_axis_angle(value):
    quat = np.asarray(value, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(
        quat[:3] * (2.0 * math.acos(quat[3]) / denominator),
        dtype=np.float32)


class LiberoAdapter(Adapter):
    contract = CONTRACT

    def __init__(self, binarize_gripper=True):
        self.binarize_gripper = bool(binarize_gripper)

    def validate(self, spec):
        self.validate_observation(spec)
        self.validate_action(spec)

    @staticmethod
    def validate_observation(spec):
        roles = {view.role for view in spec.images.views}
        if roles != {"scene", "wrist"}:
            raise ValueError(f"LIBERO requires scene/wrist roles, got {sorted(roles)}")
        if tuple(spec.state.fields) != STATE_FIELDS:
            raise ValueError("LIBERO state fields differ from PolicySpec")

    @staticmethod
    def validate_action(spec):
        action = spec.action
        if tuple(action.fields) != ACTION_FIELDS or int(action.real_dim) != 7:
            raise ValueError("LIBERO requires the frozen 7D action contract")
        if action.representation != "eef_delta_pose" or action.frame != "robot_base":
            raise ValueError("LIBERO requires robot_base eef_delta_pose actions")

    def observation(self, value, spec):
        self.validate_observation(spec)
        images = [
            {"name": "scene", "data": np.ascontiguousarray(
                value["agentview_image"][::-1, ::-1], dtype=np.uint8)},
            {"name": "wrist", "data": np.ascontiguousarray(
                value["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8)},
        ]
        state = np.concatenate((
            value["robot0_eef_pos"], quat_to_axis_angle(value["robot0_eef_quat"]),
            value["robot0_gripper_qpos"])).astype(np.float32)
        if state.shape != (8,):
            raise ValueError(f"LIBERO state must be 8D, got {state.shape}")
        return images, state

    def action(self, value, spec):
        self.validate_action(spec)
        command = np.asarray(value, dtype=np.float32).copy()
        if command.shape[-1:] != (7,):
            raise ValueError(f"LIBERO action must end in dimension 7, got {command.shape}")
        command[..., -1] = 1.0 - 2.0 * command[..., -1]
        if self.binarize_gripper:
            command[..., -1] = np.sign(command[..., -1])
        return command
