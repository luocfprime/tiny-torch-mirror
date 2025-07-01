import json
import os
import sys
from pathlib import Path

from tiny_torch_mirror.core.fetch import fetch_existing_from_local_mirror_repo


def _remote_std_interface_json():
    """Remote interface using JSON for better serialization."""
    mirror_root, packages, cuda_versions = sys.argv[1:]

    # Parse inputs
    mirror_root = Path(mirror_root)
    packages = json.loads(packages)
    cuda_versions = json.loads(cuda_versions)

    # Suppress all stdout/stderr except the final result
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        result = fetch_existing_from_local_mirror_repo(
            mirror_root, packages, cuda_versions
        )

        # Convert to JSON-serializable format
        serializable_result = [
            [wheel_name, str(wheel_path), sha256]
            for wheel_name, wheel_path, sha256 in result
        ]
    finally:
        # Restore stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Print only the JSON result
    print(json.dumps(serializable_result))
