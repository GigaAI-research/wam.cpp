from __future__ import annotations

import numpy as np

from .base import EnvironmentContract
from .libero import ACTION_FIELDS, STATE_FIELDS, LiberoAdapter

CONTRACT = EnvironmentContract(
    "liberox", ("scene", "wrist"), STATE_FIELDS, ACTION_FIELDS,
    8, 7, "eef_delta_pose", "robot_base", "continuous")


class LiberoXAdapter(LiberoAdapter):
    contract = CONTRACT

    def observation(self, value, spec):
        self.validate_observation(spec)
        images = [
            {"name": "scene", "data": np.ascontiguousarray(
                value["observation/image"], dtype=np.uint8)},
            {"name": "wrist", "data": np.ascontiguousarray(
                value["observation/wrist_image"], dtype=np.uint8)},
        ]
        state = np.ascontiguousarray(
            value["observation/state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"LIBERO-X state must be 8D, got {state.shape}")
        return images, state

    def action(self, value, spec):
        self.validate_action(spec)
        command = np.asarray(value, dtype=np.float32).copy()
        if command.ndim != 2 or command.shape[1] != 7:
            raise ValueError(f"LIBERO-X action chunk must be [T,7], got {command.shape}")
        command[:, -1] *= -1.0
        if self.binarize_gripper:
            command[:, -1] = np.sign(command[:, -1])
        return command
