import json
import re
import shlex
import subprocess
from itertools import product
from pathlib import Path
from typing import List, Set, Tuple
from urllib.parse import urljoin

import httpx
from tqdm import tqdm

from tiny_torch_mirror.core.config import get_config
from tiny_torch_mirror.core.log import logger
from tiny_torch_mirror.core.utils import (
    parse_index_page,
)


def fetch_available_from_index() -> Set[Tuple[str, str, str]]:
    """
    Fetch available wheels from the index (source).

    :return: A set of tuples containing (wheel_name, wheel_url, wheel_sha256).
    """
    config = get_config()
    wheels = set()
    pbar = tqdm(
        desc=f"Fetching available wheels from index [0]",
        total=len(config.packages) * len(config.cuda_versions),
    )

    for package, cuda_version in product(config.packages, config.cuda_versions):
        index_url = urljoin(config.index_base_url, f"/whl/{cuda_version}/{package}")
        try:
            response = httpx.get(index_url)
            new_wheels = parse_index_page(
                response.text,
                config.index_base_url,
            )

            # filter out those with link pointing to non-cuda wheels
            # e.g. /whl/cu118/torchvision-0.15.1-cp310-cp310-manylinux2014_aarch64.whl
            # is pointing to https://download.pytorch.org/whl/torchvision-0.15.1-cp310-cp...
            new_wheels = filter(lambda whl: re.search(r"cu\d+", whl[1]), new_wheels)

            wheels.update(set(new_wheels))

            pbar.set_description(
                desc=f"Fetching available wheels from index [{len(wheels)}]"
            )
            pbar.update(1)

            if not new_wheels:
                logger.warning(f"No wheels found for {package} {cuda_version}")

        except Exception:
            logger.error(f"Failed to fetch index {index_url}")

    pbar.close()

    return wheels


def fetch_existing_from_local_mirror_repo(
    mirror_root: Path, packages: List[str], cuda_versions: List[str]
) -> Set[Tuple[str, str, str]]:
    """
    Fetch existing wheels from local mirror repo (dest).

    :return: A set of tuples containing (wheel_name, wheel_path, wheel_sha256).
    """
    wheels = set()
    pbar = tqdm(
        desc=f"Fetching existing wheels from mirror [0]",
        total=len(packages) * len(cuda_versions),
    )

    for package, cuda_version in product(packages, cuda_versions):
        index_directory = Path(mirror_root, "whl", cuda_version, package)

        if not index_directory.exists():
            logger.debug(f"{index_directory} does not exist.")
            continue

        try:
            actual_wheel_names = [
                wheel_path.name for wheel_path in index_directory.glob("*.whl")
            ]
        except Exception:
            logger.warning(f"{index_directory} could be missing or inaccessible.")
            continue

        logger.debug(f"{index_directory} contains {len(actual_wheel_names)} wheels.")

        try:
            index_wheels = parse_index_page(  # wheels that are in the index
                (index_directory / "index.html").read_text(),
                base_url="/",
            )
        except Exception:
            logger.warning(
                f"{index_directory}/index.html could be missing or inaccessible."
            )
            continue

        logger.debug(f"{index_directory} index contains {len(index_wheels)} wheels.")

        new_wheels = {
            (
                wheel_name,
                index_directory / wheel_name,
                sha256,
            )
            for wheel_name, _, sha256 in index_wheels
            if wheel_name in actual_wheel_names
        }

        logger.debug(
            f"index_wheels={index_wheels}, new_wheels={new_wheels}, actual_wheel_names={actual_wheel_names}"
        )

        wheels.update(new_wheels)

        pbar.set_description(
            desc=f"Fetching existing wheels from mirror [{len(wheels)}]"
        )
        pbar.update(1)

        if not new_wheels:
            logger.warning(f"No wheels found for {package} {cuda_version}")

    pbar.close()

    return wheels


def fetch_existing_from_remote_mirror_repo() -> Set[Tuple[str, str, str]]:
    """
    Fetch existing wheels from remote mirror repo via SSH using JSON.
    """
    config = get_config()
    ssh_config = config.ssh

    # Convert to JSON strings and properly escape for shell
    packages_json = shlex.quote(json.dumps(config.packages))
    cuda_versions_json = shlex.quote(json.dumps(config.cuda_versions))
    mirror_root = shlex.quote(str(config.mirror_root))

    # Build remote command
    remote_command = f"_tiny_torch_mirror_remote_std_interface_json {mirror_root} {packages_json} {cuda_versions_json}"

    cmd = [
        "ssh",
        f"{ssh_config.user}@{ssh_config.ip}",
        "-p",
        str(ssh_config.port),
        remote_command,
    ]

    logger.debug(f"Running SSH command: {' '.join(cmd)}")

    try:
        # Run command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minute timeout
        )

        output = result.stdout.strip()

        if not output:
            logger.warning("Received empty response from remote mirror")
            return set()

        logger.debug(f"Received remote JSON output: {output[:1000]}")

        try:
            result_list = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse remote JSON output: {e}")
            logger.debug(f"Raw output: {output[:1000]}")  # First 1000 chars
            raise RuntimeError(f"Invalid JSON response from remote mirror: {e}")

        # Validate the structure
        if not isinstance(result_list, list):
            raise ValueError(f"Expected list, got {type(result_list).__name__}")

        # Convert to set of tuples with validation
        wheels = set()
        for item in result_list:
            if not isinstance(item, list) or len(item) != 3:
                logger.warning(f"Skipping invalid wheel entry: {item}")
                continue
            wheel_name, wheel_path, sha256 = item
            if all(isinstance(x, str) for x in [wheel_name, wheel_path, sha256]):
                wheels.add((wheel_name, wheel_path, sha256))
            else:
                logger.warning(f"Skipping wheel with non-string values: {item}")

        logger.info(f"Found {len(wheels)} existing wheels on remote mirror")
        return wheels

    except subprocess.TimeoutExpired:
        logger.error("SSH command timed out")
        raise RuntimeError("Remote mirror fetch timed out after 5 minutes")

    except subprocess.CalledProcessError as e:
        error_message = e.stderr if e.stderr else str(e)
        logger.error(f"SSH command failed: {error_message}")
        logger.debug(f"Exit code: {e.returncode}")
        raise RuntimeError(
            f"Failed to fetch existing wheels from remote mirror: {error_message}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        raise
