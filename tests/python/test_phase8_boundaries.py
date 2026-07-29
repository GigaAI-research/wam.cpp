from pathlib import Path


def run():
    root = Path(__file__).resolve().parents[2]
    proto = (root / "proto/wam.proto").read_text()
    assert "package wam.rpc.v06;" in proto
    assert "artifact_sha256" not in proto
    production = [
        *root.glob("python/wam/**/*.py"),
        *root.glob("eval/**/*.py"),
    ]
    text = "\n".join(path.read_text() for path in production)
    assert "wam.rpc.v05" not in text
    assert "artifact_sha256" not in text
    assert "architecture ==" not in text and "architecture in (" not in text

    assert not any((root / "eval/common").glob("*.py"))

    for name in ("run_libero_client.py", "run_liberox_client.py"):
        value = (root / "eval/sim" / name).read_text()
        for function in (
                "def observation_to_policy_observation",
                "def policy_action_to_command",
                "def check_observation_compatibility",
                "def check_action_compatibility"):
            assert function not in value
    for name in ("run_robotwin_server.py", "run_libero_server.py",
                 "run_liberox_server.py"):
        value = (root / "eval/sim" / name).read_text()
        assert "EnvironmentContract(" not in value
        assert "from wam.adapters" in value
        assert "from wam.serving" in value


if __name__ == "__main__":
    run()
    print("wam Phase 8 source boundaries: PASS")
