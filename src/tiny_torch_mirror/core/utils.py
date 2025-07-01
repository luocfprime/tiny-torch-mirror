import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin

from packaging.utils import parse_wheel_filename

from tiny_torch_mirror.core.config import get_config


def parse_wheel_name(wheel_name: str) -> Dict[str, str]:
    """
    Parses a wheel filename into its components using the `packaging` library.

    :param wheel_name: The wheel filename (e.g., 'torchvision-0.14.0+cu116-cp39-cp39-linux_x86_64.whl')
    :return: Dictionary containing parsed components:
             - package_name: Name of the package
             - version: Version of the package (without local identifier)
             - python_version: Python version (e.g., 'cp39')
             - platform: Operating system and architecture (e.g., 'linux_x86_64')
    """
    try:
        # Use the standard `parse_wheel_filename` for robust parsing.
        # This correctly separates the name, version, build tag, and platform tags.
        name, version_obj, build, tags = parse_wheel_filename(wheel_name)

        # The version object's string representation includes the local version identifier (e.g., '1.13.0+cu117').
        # We split this to separate the base version from the local part.
        version_parts = str(version_obj).split("+", 1)
        base_version = version_parts[0]
        local_version = version_parts[1] if len(version_parts) > 1 else None

        # The first tag in the set usually provides the most specific platform info.
        tag = next(iter(tags))

        result = {
            "package_name": name,
            "version": base_version,
            "python_version": tag.interpreter,
            "platform": tag.platform,
        }

        # # Check the local version part to see if it specifies a CUDA version.
        # # We use re.fullmatch to ensure the local version is *only* a CUDA string
        # # (e.g., 'cu117') and not something more complex (e.g., 'cu117.with.pypi.cudnn').
        # cuda_match = re.fullmatch(r"(cu\d+)", local_version or "")
        # assert (
        #     cuda_match
        # ), f"Invalid local version: '{local_version}' in wheel '{wheel_name}' which does not contain a valid CUDA version."
        # result["cuda_version"] = cuda_match.group(1)

        return result

    except Exception as e:
        raise ValueError(f"Could not parse wheel name: '{wheel_name}'. Error: {e}")


def parse_index_page(html_content: str, base_url: str) -> List[Tuple[str, str, str]]:
    """
    Parses the HTML content of the PyTorch package index page and returns a list of available wheels.

    :param html_content: The HTML content of the index page
    :param base_url: The base URL of the index page, used to construct full URLs for wheels
    :return: List of tuples containing the wheel name, url and sha256
    """

    class WheelParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.wheels = []
            self.in_anchor = False
            self.current_url = ""
            self.current_name = ""

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.in_anchor = True
                attrs_dict = dict(attrs)
                if "href" in attrs_dict:
                    href = attrs_dict["href"]
                    # Split URL and SHA256 hash if present
                    if "#sha256=" in href:
                        url_part, sha256_part = href.split("#sha256=", 1)
                        self.current_url = urljoin(base_url, url_part)
                        self.current_sha256 = sha256_part
                    else:
                        self.current_url = urljoin(base_url, href)
                        self.current_sha256 = ""

        def handle_endtag(self, tag):
            if tag == "a":
                self.in_anchor = False

        def handle_data(self, data):
            if self.in_anchor and data.strip():
                # Get the wheel name from the text content of the anchor tag
                self.current_name = data.strip()
                if self.current_name.endswith(".whl"):
                    # Store wheel information as a tuple (name, url, sha256)
                    self.wheels.append(
                        (self.current_name, self.current_url, self.current_sha256)
                    )

    # Create and use the HTML parser
    parser = WheelParser()
    parser.feed(html_content)
    return parser.wheels


def get_wheel_dest_path(wheel_name: str, cuda_version: str) -> Path:
    """
    Maps a wheel name to a local file path based on the current working directory.

    :param wheel_name: The wheel filename (e.g., 'torchvision-0.14.0+cu116-cp39-cp39-linux_x86_64.whl')
    :param cuda_version: The CUDA version (e.g., 'cu117', 'cu118', etc.)
    :return: Path object representing the remote file path on the mirror server. (e.g. "/remote/mirror/cu117/torch/")
    """
    wheel_info = parse_wheel_name(wheel_name)
    config = get_config()

    return Path(
        config.mirror_root,
        "whl",
        cuda_version,
        wheel_info["package_name"],
        wheel_name,
    )


def remote_path_exists(remote_path: Path) -> bool:
    """
    Check if a remote path exists using SSH.
    """
    ssh_config = get_config().ssh

    # Use test command to check if the path exists
    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        f"test -e {remote_path} && echo 'exists' || echo 'not exists'",
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode().strip()
        return output == "exists"
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to check remote path existence: {error_message}")


def remote_ls(remote_path: Path) -> List[str]:
    """
    List files in a remote directory using SSH.
    """
    ssh_config = get_config().ssh

    # Use ls command to list files and directories
    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        f"ls -1 {remote_path}",  # -1 option outputs one file per line
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode().strip()
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to list remote directory: {error_message}")

    if output:
        return output.split("\n")
    return []


def remote_mkdir(remote_path: Path) -> None:
    """
    Create a remote directory using SSH.
    """
    ssh_config = get_config().ssh

    # Use mkdir command to create the directory
    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        f"mkdir -p {remote_path}",  # -p option creates parent directories as needed
    ]

    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to create remote directory: {error_message}")


def remote_cat(remote_path: Path) -> str:
    """
    Cat the file from a remote directory using SSH.
    """
    ssh_config = get_config().ssh

    # Use cat command to read file
    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        f"cat {remote_path}",
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode()
        return output
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to read remote file: {error_message}")


def remote_update_index_html(
    remote_path: Path, wheels: List[Tuple[str, str, str]]
) -> None:
    """
    Update the index.html file in a remote directory with the provided wheel information.
    """
    ssh_config = get_config().ssh

    # Create the index.html content
    index_content = "<html><body>\n"
    for wheel_name, wheel_url, wheel_sha256 in wheels:
        if wheel_sha256:
            index_content += (
                f'<a href="{wheel_url}#sha256={wheel_sha256}">{wheel_name}</a><br>\n'
            )
        else:
            index_content += f'<a href="{wheel_url}">{wheel_name}</a><br>\n'

    index_content += "</body></html>"

    # Use echo command to write the content to index.html
    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        f"echo -e '{index_content}' > {remote_path}",
    ]

    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to update remote index.html: {error_message}")
