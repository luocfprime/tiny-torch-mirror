import getpass
from pathlib import Path
from typing import List

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class SSHConfig(BaseSettings):
    ip: str = Field("localhost", description="SSH server IP address or hostname")
    port: int = Field(22, description="SSH server port")
    user: str = Field(getpass.getuser(), description="SSH username for authentication")


class PyTorchMirrorConfig(BaseSettings):
    ssh: SSHConfig = Field(
        SSHConfig(),  # type: ignore[missing-call-arg]
        description="SSH connection configuration",
    )  # noqa
    mirror_root: str = Field(
        "~/pytorch_mirror",
        description="Path on the server where PyTorch packages will be mirrored",
    )
    local_cache_dir: str = Field(
        "/tmp/pytorch_mirror",
        description="Local directory for temporary storage of downloaded packages",
    )
    index_base_url: str = Field(
        "https://download.pytorch.org",
        description="Base URL for the PyTorch package index",
    )
    platforms: List[str] = Field(
        ["linux_x86_64"],
        description="List of platforms to support (e.g., linux_x86_64). ",
    )
    python_versions: List[str] = Field(
        ["cp38", "cp39", "cp310", "cp311", "cp312"],
        description="List of Python versions to support (format: cp38, cp39, cp310, etc.)",
    )
    cuda_versions: List[str] = Field(
        ["cu117", "cu118", "cu121"],
        description="List of CUDA versions to support (format: cu117, cu118, etc.)",
    )
    packages: List[str] = Field(
        ["torch", "torchvision", "torchaudio", "xformers"],
        description="List of PyTorch-related packages to mirror",
    )

    @field_validator("index_base_url")
    def normalize_url(cls, v: str, field) -> str:
        if not v.endswith("/"):
            v += "/"
        return v


_config = None

CONFIG_PATH = Path("pytorch_mirror_config.yml")


def load_config(config_path: str | Path) -> PyTorchMirrorConfig:
    """
    Load configuration from a YAML file.
    """
    global _config

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    _config = PyTorchMirrorConfig(**config_data)

    return _config


def get_config() -> PyTorchMirrorConfig:
    """
    Get the loaded configuration.
    """
    if _config is None:
        raise ValueError("Configuration not loaded. Call load_config() first.")
    return _config  # noqa
