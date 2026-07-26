"""Run wam.cpp with the frozen RoboTwin dual-arm environment contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.server import EnvironmentContract, serve_main


JOINT_FIELDS = tuple(
    f"{arm}.joint.{index}" if index < 6 else f"{arm}.gripper"
    for arm in ("left", "right") for index in range(7)
)

CONTRACT = EnvironmentContract(
    environment_id="robotwin",
    image_roles=("camera_high", "camera_left_wrist", "camera_right_wrist"),
    state_fields=JOINT_FIELDS,
    action_fields=JOINT_FIELDS,
    state_dim=14,
    action_dim=14,
    action_representation="joint_position",
    action_frame="controller",
    gripper_encoding="continuous",
)


if __name__ == "__main__":
    serve_main(CONTRACT)
