import json
import subprocess
import sys


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "juniper_encoder", *args], capture_output=True, text=True)


def test_help_is_available():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "Juniper Encoder" in result.stdout


def test_blocked_environment_is_machine_readable():
    result = run_cli("env", "verify", "--lock", "requirements/flowbox.lock")
    payload = json.loads(result.stdout)
    if result.returncode == 0:
        assert payload["status"] == "PASS"
        assert payload["cuda_available"] is True
    else:
        assert payload["error"] == "BLOCKED_ENVIRONMENT"
