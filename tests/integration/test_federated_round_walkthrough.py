"""Run the federated walkthrough with network and framework imports blocked."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_RUNNER = """
import importlib.abc
import runpy
import socket
import sys

blocked = {
    'torch', 'transformers', 'peft', 'accelerate', 'mlx', 'tensorflow',
    'jax', 'numpy', 'huggingface_hub', 'datasets', 'flwr',
}
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('optional framework import attempted')
sys.meta_path.insert(0, BlockFrameworks())

def deny_network(*args, **kwargs):
    raise AssertionError('network access attempted')
class OfflineSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        return deny_network(*args, **kwargs)
socket.socket = OfflineSocket
socket.create_connection = deny_network
socket.getaddrinfo = deny_network

# Negative controls confirm both guards are active before importing the example.
for probe in (lambda: __import__('torch'), lambda: socket.create_connection(('localhost', 1))):
    try:
        probe()
    except AssertionError:
        pass
    else:
        raise AssertionError('offline guard was bypassed')

output_format = sys.argv[1]
sys.argv = ['federated_round_walkthrough', '--format', output_format]
runpy.run_module('examples.federated_round_walkthrough', run_name='__main__')
"""


@pytest.mark.integration
@pytest.mark.parametrize("output_format", ("json", "markdown"))
def test_entry_point_is_offline_framework_free_and_byte_stable(output_format):
    extension = "json" if output_format == "json" else "md"
    expected = (
        ROOT / f"tests/fixtures/training/federated_round_walkthrough.{extension}"
    ).read_bytes()
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", OFFLINE_RUNNER, output_format],
            cwd=ROOT,
            capture_output=True,
            timeout=5,
            check=True,
        )
        assert completed.stderr == b""
        assert completed.stdout == expected
        assert completed.stdout.endswith(b"\n")
        assert not completed.stdout.endswith(b"\n\n")
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
