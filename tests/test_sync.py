import getpass
from pathlib import Path

import pytest

from tiny_torch_mirror.core.config import PyTorchMirrorConfig, SSHConfig
from tiny_torch_mirror.core.sync import rsync


@pytest.fixture(autouse=True)
def setup_ssh_config(monkeypatch):
    new_config = PyTorchMirrorConfig(
        ssh=SSHConfig(ip="localhost", port=22, user=getpass.getuser()),
        mirror_root="/tmp/test_pytorch_mirror",
        local_cache_dir="/tmp/test_pytorch_mirror_cache",
    )  # noqa
    monkeypatch.setattr("tiny_torch_mirror.core.config._config", new_config)


def test_rsync():
    src_file = Path("/tmp/test_pytorch_mirror/test_file.txt")
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text("test")

    remote_dir = Path("/tmp/test_pytorch_mirror_remote")
    remote_dir.mkdir(parents=True, exist_ok=True)

    rsync(src_file, remote_dir)

    assert (remote_dir / src_file.name).exists()
    assert (remote_dir / src_file.name).read_text() == "test"
