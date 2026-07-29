from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentContract:
    environment_id: str
    image_roles: tuple[str, ...]
    state_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    state_dim: int
    action_dim: int
    action_representation: str
    action_frame: str
    gripper_encoding: str


def check_environment(metadata, contract):
    spec = metadata["policy_spec"]
    roles = tuple(view["role"] for view in spec["images"]["views"])
    failures = []
    checks = (
        (set(roles) == set(contract.image_roles),
         f"image roles {roles} != {contract.image_roles}"),
        (int(spec["state"]["real_dim"]) == contract.state_dim,
         f"state dim {spec['state']['real_dim']} != {contract.state_dim}"),
        (tuple(spec["state"]["fields"]) == contract.state_fields,
         "state field order differs from environment contract"),
        (int(spec["action"]["real_dim"]) == contract.action_dim,
         f"action dim {spec['action']['real_dim']} != {contract.action_dim}"),
        (spec["action"]["representation"] == contract.action_representation,
         "action representation differs from environment contract"),
        (spec["action"]["frame"] == contract.action_frame,
         "action frame differs from environment contract"),
        (tuple(spec["action"]["fields"]) == contract.action_fields,
         "action field order differs from environment contract"),
        (spec["action"]["gripper"] == contract.gripper_encoding,
         "gripper encoding differs from environment contract"),
    )
    failures.extend(message for valid, message in checks if not valid)
    if failures:
        raise ValueError(
            f"{contract.environment_id} contract incompatible: " +
            "; ".join(failures))


class Adapter:
    contract: EnvironmentContract

    def validate(self, policy_spec):
        raise NotImplementedError

    def observation(self, value, policy_spec):
        raise NotImplementedError

    def action(self, value, policy_spec):
        raise NotImplementedError
