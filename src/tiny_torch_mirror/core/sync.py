import subprocess
from pathlib import Path

from tiny_torch_mirror.core.config import get_config


def rsync(local_file_path: Path, remote_file_path: Path):
    config = get_config()
    remote_dest = f"{config.ssh.user}@{config.ssh.ip}:{remote_file_path.absolute()}"

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
    ]

    rsync_cmd.extend(["-e", f"ssh -p {config.ssh.port}"])
    rsync_cmd.extend([local_file_path, remote_dest])

    subprocess.run(rsync_cmd, check=True)
