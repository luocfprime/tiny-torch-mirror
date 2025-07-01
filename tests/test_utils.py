import getpass
import subprocess
from pathlib import Path
from shutil import rmtree

import pytest

from tiny_torch_mirror.core.config import PyTorchMirrorConfig, SSHConfig, get_config
from tiny_torch_mirror.core.utils import (
    parse_index_page,
    parse_wheel_name,
    remote_cat,
    remote_ls,
    remote_mkdir,
    remote_update_index_html,
)


class TestParseWheelName:

    def test_simple_torch_with_cuda(self):
        result = parse_wheel_name("torch-1.13.0+cu117-cp310-cp310-linux_x86_64.whl")
        assert result == {
            "package_name": "torch",
            "version": "1.13.0",
            # "cuda_version": "cu117",
            "python_version": "cp310",
            "platform": "linux_x86_64",
        }

    def test_xformers_with_post_version(self):
        result = parse_wheel_name(
            "xformers-0.0.22.post7+cu118-cp311-cp311-manylinux2014_x86_64.whl"
        )
        assert result == {
            "package_name": "xformers",
            "version": "0.0.22.post7",
            # "cuda_version": "cu118",
            "python_version": "cp311",
            "platform": "manylinux2014_x86_64",
        }

    def test_torch_windows_platform(self):
        result = parse_wheel_name("torch-1.13.0+cu117-cp39-cp39-win_amd64.whl")
        assert result == {
            "package_name": "torch",
            "version": "1.13.0",
            # "cuda_version": "cu117",
            "python_version": "cp39",
            "platform": "win_amd64",
        }

    def test_python37m(self):
        result = parse_wheel_name("torch-1.13.0+cu117-cp37-cp37m-linux_x86_64.whl")
        assert result == {
            "package_name": "torch",
            "version": "1.13.0",
            # "cuda_version": "cu117",
            "python_version": "cp37",
            "platform": "linux_x86_64",
        }

    def test_invalid_wheel_name(self):
        with pytest.raises(ValueError):
            parse_wheel_name("not-a-valid-wheel-file.whl")

    def test_missing_extension(self):
        with pytest.raises(ValueError):
            parse_wheel_name("torch-1.13.0+cu117-cp39-cp39-linux_x86_64")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://download.pytorch.org/whl/cu117",
        "https://download.pytorch.org/whl/cu117/",
    ],
)
def test_parse_index_page(base_url):
    html_content = """
    <!DOCTYPE html>
    <html>
      <body>
        <h1>Links for torch</h1>
        <a href="/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-win_amd64.whl#sha256=a0b87b3c87e16f472aa5c87dd31a071211cdec972de280ad1aacff0d245354f8">torch-1.13.0+cu117-cp310-cp310-win_amd64.whl</a><br/>
        <a href="/whl/cu117/torch-1.13.0%2Bcu117-cp37-cp37m-linux_x86_64.whl#sha256=eaecf21f4944f62302f73a6a69f0a8fa3f25ad3cff536b1eef5a5cf4203b6fd0">torch-1.13.0+cu117-cp37-cp37m-linux_x86_64.whl</a><br/>
        <a href="/whl/cu117/torch-1.13.0%2Bcu117-cp37-cp37m-win_amd64.whl#sha256=08a059d4f298ed042c541c326969004252b0a106a960150163587560782879b4">torch-1.13.0+cu117-cp37-cp37m-win_amd64.whl</a><br/>
        <a href="/whl/torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl#sha256=0882243755ff28895e8e6dc6bc26ebcf5aa0911ed81b2a12f241fc4b09075b13">torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl</a><br/>
        <a href="/whl/torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl#sha256=423e0ae257b756bb45a4b49072046772d1ad0c592265c5080070e0767da4e490">torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl</a><br/>
      </body>
    </html>
    <!--TIMESTAMP 1751116657-->
    """
    results = parse_index_page(html_content, base_url)

    assert results == [
        (
            "torch-1.13.0+cu117-cp310-cp310-win_amd64.whl",
            "https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-win_amd64.whl",
            "a0b87b3c87e16f472aa5c87dd31a071211cdec972de280ad1aacff0d245354f8",
        ),
        (
            "torch-1.13.0+cu117-cp37-cp37m-linux_x86_64.whl",
            "https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp37-cp37m-linux_x86_64.whl",
            "eaecf21f4944f62302f73a6a69f0a8fa3f25ad3cff536b1eef5a5cf4203b6fd0",
        ),
        (
            "torch-1.13.0+cu117-cp37-cp37m-win_amd64.whl",
            "https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp37-cp37m-win_amd64.whl",
            "08a059d4f298ed042c541c326969004252b0a106a960150163587560782879b4",
        ),
        (
            "torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl",
            "https://download.pytorch.org/whl/torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl",
            "0882243755ff28895e8e6dc6bc26ebcf5aa0911ed81b2a12f241fc4b09075b13",
        ),
        (
            "torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl",
            "https://download.pytorch.org/whl/torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl",
            "423e0ae257b756bb45a4b49072046772d1ad0c592265c5080070e0767da4e490",
        ),
    ]


class TestRemoteUtils:
    @pytest.fixture(autouse=True)
    def setup_ssh_config(self, monkeypatch):
        new_config = PyTorchMirrorConfig(
            ssh=SSHConfig(ip="localhost", port=22, user=getpass.getuser()),
            mirror_root="/tmp/test_pytorch_mirror",
            local_cache_dir="/tmp/test_pytorch_mirror_cache",
        )  # noqa
        monkeypatch.setattr("tiny_torch_mirror.core.config._config", new_config)

    def test_config_fixture(self):
        config = get_config()
        assert config.ssh.ip == "localhost"
        assert config.ssh.user == getpass.getuser()

    def test_local_ssh_connection(self):
        try:
            ssh_config = get_config().ssh
            subprocess.check_output(
                [
                    "ssh",
                    f"{ssh_config.user}@{ssh_config.ip}",
                    "-p",
                    str(ssh_config.port),
                    "echo",
                    "Hello, World!",
                ]
            )
        except subprocess.CalledProcessError as e:
            raise AssertionError(f"SSH connection failed. Error: {e.stderr.decode()}")

    def test_remote_mkdir(self):
        test_mkdir_path = Path("/tmp/test_pytorch_mirror/test_dir")
        try:
            rmtree(test_mkdir_path)
        except FileNotFoundError:
            pass

        remote_mkdir(test_mkdir_path)

        assert test_mkdir_path.exists()

        rmtree(test_mkdir_path)  # Clean up after test

    def test_remote_ls(self):
        test_ls_path = Path("/tmp/test_pytorch_mirror/test_dir")
        test_ls_path.mkdir(parents=True, exist_ok=True)

        # create a bunch of dummy files
        (test_ls_path / "file1.txt").touch()
        (test_ls_path / "file2.txt").touch()

        result = remote_ls(test_ls_path)

        assert "file1.txt" in result
        assert "file2.txt" in result

        assert len(result) == 2

        rmtree(test_ls_path)

    def test_remote_cat(self):
        test_index_path = Path("/tmp/test_pytorch_mirror/index.html")
        test_index_path.parent.mkdir(parents=True, exist_ok=True)

        content = """
            <!DOCTYPE html>
            <html>
              <body>
                <h1>Links for torch</h1>
                <a href="/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl#sha256=bbf9546f0d0d8b51263ca479637b426a88335fca0034f42cec63d4d32dee05af">torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl</a><br/>
              </body>
            </html>
            """

        test_index_path.write_text(content)

        result = remote_cat(test_index_path)

        assert result == content

    def test_remote_update_index_html(self):
        test_index_path = Path("/tmp/test_pytorch_mirror/index.html")
        test_index_path.parent.mkdir(parents=True, exist_ok=True)

        content = """
            <!DOCTYPE html>
            <html>
              <body>
                <h1>Links for torch</h1>
                <a href="/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl#sha256=bbf9546f0d0d8b51263ca479637b426a88335fca0034f42cec63d4d32dee05af">torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl</a><br/>
              </body>
            </html>
            """
        test_index_path.write_text(content)

        wheels = parse_index_page(content, "/")
        wheels.extend(
            [
                (
                    "torch-2.1.0+cu121-cp310-cp310-win_amd64.whl",
                    # "https://download.pytorch.org/whl/cu121/torch-2.1.0+cu121-cp310-cp310-win_amd64.whl",
                    "/whl/cu121/torch-2.1.0+cu121-cp310-cp310-win_amd64.whl",
                    "6ee083ba804e863af059ea284c1678c1b0628699fb0014c8e043ceed7d4ce930",
                ),
                (
                    "torch-2.1.0+cu121-cp39-cp39-linux_x86_64.whl",
                    # "https://download.pytorch.org/whl/cu121/torch-2.1.0+cu121-cp39-cp39-linux_x86_64.whl",
                    "/whl/cu121/torch-2.1.0+cu121-cp39-cp39-linux_x86_64.whl",
                    "94b60ae7562ae732554ae8744123b33d46e659c3251a5a58c7269c12e838868b",
                ),
            ]
        )

        remote_update_index_html(test_index_path, wheels)

        result = parse_index_page(test_index_path.read_text(), "/")

        for name, url, sha256 in wheels:
            assert (name, url, sha256) in result

        assert len(result) == len(wheels)
