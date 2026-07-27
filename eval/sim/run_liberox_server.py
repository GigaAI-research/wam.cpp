"""Run wam.cpp with the frozen LIBERO-X end-effector contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.server import EnvironmentContract, serve_main


CONTRACT = EnvironmentContract(
    environment_id="liberox",
    image_roles=("scene", "wrist"),
    state_fields=(
        "eef.position.x", "eef.position.y", "eef.position.z",
        "eef.rotation.axis_angle.x", "eef.rotation.axis_angle.y",
        "eef.rotation.axis_angle.z", "gripper.left", "gripper.right"),
    action_fields=(
        "eef.delta.position.x", "eef.delta.position.y",
        "eef.delta.position.z", "eef.delta.rotation.axis_angle.x",
        "eef.delta.rotation.axis_angle.y",
        "eef.delta.rotation.axis_angle.z", "gripper.command"),
    state_dim=8,
    action_dim=7,
    action_representation="eef_delta_pose",
    action_frame="robot_base",
    gripper_encoding="continuous",
)


if __name__ == "__main__":
    serve_main(CONTRACT)
