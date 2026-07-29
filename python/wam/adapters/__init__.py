from .base import Adapter, EnvironmentContract, check_environment
from .libero import LiberoAdapter
from .liberox import LiberoXAdapter
from .robotwin import RoboTwinAdapter

ADAPTERS = {
    "robotwin": RoboTwinAdapter,
    "libero": LiberoAdapter,
    "liberox": LiberoXAdapter,
}


def create_adapter(environment_id, **kwargs):
    try:
        return ADAPTERS[environment_id](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown environment adapter: {environment_id}") from error


__all__ = ["Adapter", "EnvironmentContract", "LiberoAdapter",
           "LiberoXAdapter", "RoboTwinAdapter", "check_environment",
           "create_adapter"]
