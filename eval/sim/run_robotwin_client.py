"""Run the RoboTwin simulator client against a wam.cpp server."""


def check_observation_compatibility(policy_spec):
    raise NotImplementedError


def observation_to_policy_observation(observation, policy_spec):
    raise NotImplementedError


def check_action_compatibility(policy_spec):
    raise NotImplementedError


def policy_action_to_command(action, policy_spec):
    raise NotImplementedError


def main(argv=None):
    raise NotImplementedError


if __name__ == "__main__":
    main()
