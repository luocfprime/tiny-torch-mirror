import hashlib
import itertools
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
import uvicorn
import yaml
from rich.console import Console
from tqdm import tqdm

from tiny_torch_mirror.core.config import (
    CONFIG_PATH,
    PyTorchMirrorConfig,
    load_config,
)
from tiny_torch_mirror.core.fetch import (
    fetch_available_from_index,
    fetch_existing_from_local_mirror_repo,
    fetch_existing_from_remote_mirror_repo,
)
from tiny_torch_mirror.core.job import run_jobs_in_threadpool
from tiny_torch_mirror.core.ui import PackageViewerApp
from tiny_torch_mirror.core.utils import parse_wheel_name

from .core.server import create_app

logging.basicConfig(level=logging.INFO)

console = Console()
app = typer.Typer()


@app.command()
def config(config_path: Path = CONFIG_PATH):
    """Initialize config. (Run this on local machine where network is available)"""
    if not config_path.exists():
        typer.confirm("No configuration file found. Create one?", abort=True)
        config_path.write_text(yaml.dump(PyTorchMirrorConfig().model_dump()))  # type: ignore[call-arg]

    print(
        f"Config file is at {config_path.absolute()}. To edit it, run \n```bash\nvim {config_path.absolute()}\n```"
    )


@app.command()
def sync(config_path: Path = CONFIG_PATH):
    """Update the remote mirror repo to sync with PyTorch index. (Run this on local machine where network is available)"""
    config = load_config(config_path)

    available_wheels = fetch_available_from_index()
    existing_wheels = fetch_existing_from_remote_mirror_repo()

    # wheel: (wheel_name, wheel_url, wheel_sha256)
    available_wheels_dict = {
        wheel[0]: wheel for wheel in available_wheels  # key: wheel name
    }
    existing_wheels_dict = {wheel[0]: wheel for wheel in existing_wheels}

    # Organize wheels by package
    packages = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    platforms = (
        list(itertools.chain(*config.platforms.values()))
        if isinstance(config.platforms, dict)
        else config.platforms
    )

    for wheel_name in set(
        list(available_wheels_dict.keys()) + list(existing_wheels_dict.keys())
    ):
        try:
            parsed = parse_wheel_name(wheel_name)
        except ValueError:
            continue

        package_name = parsed["package_name"]
        version = parsed["version"]
        py_ver = parsed["python_version"]
        platform = parsed["platform"]

        # get cuda version from url part (since packages like xformers do not have cuda version in the wheel name in
        # newer versions)
        wheel = available_wheels_dict.get(wheel_name) or existing_wheels_dict.get(
            wheel_name
        )
        _, url, _ = wheel
        cuda_ver = re.search(r"cu\d+", url).group(0)

        if not (
            package_name in config.packages
            and py_ver in config.python_versions
            and platform in platforms
            and cuda_ver in config.cuda_versions
        ):
            continue

        variant = f"{cuda_ver}+{py_ver}+{platform}"
        packages[package_name][variant][version] = {
            "available": wheel_name in available_wheels_dict,
            "installed": wheel_name in existing_wheels_dict,
            "wheel_name": wheel_name,
        }

    if not packages:
        console.print("[yellow]No packages match the configuration criteria.[/yellow]")
        return

    # Launch TUI application
    app_instance = PackageViewerApp(packages, available_wheels_dict)
    app_instance.run()

    # Check if user confirmed the update
    if not app_instance.confirmed:
        console.print("\n[yellow]Update cancelled.[/yellow]")
        return

    to_be_updated = list(app_instance.all_to_be_updated)

    if not to_be_updated:
        console.print("\n[green]✓ Mirror is up to date![/green]")
        return

    # Show final confirmation
    console.print(
        f"\n[bold]Preparing to download {len(to_be_updated)} wheels...[/bold]"
    )

    # Download wheels and update mirror repo
    jobs = [
        (wheel_name, download_url, sha256)
        for wheel_name, download_url, sha256 in available_wheels
        if wheel_name in to_be_updated
    ]

    run_jobs_in_threadpool(jobs)

    console.print(f"\n[green]✓ Successfully downloaded {len(jobs)} wheels![/green]")


@app.command()
def serve(
    path: str = typer.Option("~/pytorch_mirror", help="Path to the mirror root"),
    port: int = typer.Option(8080, help="Port to serve the mirror on"),
    fallback_index: str = typer.Option(
        "https://pypi.org/simple/", help="Fallback index URL for packages not in mirror"
    ),
    path_map: Optional[str] = typer.Option(
        None,
        help='JSON string mapping local paths to fallback paths, e.g., \'{"whl/cu118": ""}\'',
    ),
):

    mirror_path = Path(path).expanduser().resolve()

    # Parse path mappings
    path_mappings = {}
    if path_map:
        try:
            path_mappings = json.loads(path_map)
        except json.JSONDecodeError:
            print(f"Warning: Invalid path_map JSON: {path_map}")

    # Create FastAPI app
    app = create_app(mirror_path, fallback_index, path_mappings)

    # Print startup info
    print(f"\n{'=' * 60}")
    print(f"PyTorch Mirror Server")
    print(f"{'=' * 60}")
    print(f"Serving at: http://0.0.0.0:{port}")
    print(f"Mirror path: {mirror_path}")
    print(f"Fallback to: {fallback_index}")

    if path_mappings:
        print(f"\nPath mappings:")
        for local, remote in path_mappings.items():
            print(f"  {local} -> {remote or '(root)'}")

    print(f"\nExample usage:")
    print(f"  pip install torch --index-url http://localhost:{port}/whl/cu118")
    print(f"\n{'=' * 60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)


@app.command()
def verify(config_path: Path = CONFIG_PATH):
    """Verify the integrity of the mirror repo."""
    config = load_config(config_path)
    wheels = fetch_existing_from_local_mirror_repo(
        Path(config.mirror_root), config.packages, config.cuda_versions
    )

    def verify_wheel(wheel_info):
        name, path, expected_sha = wheel_info

        if not expected_sha:
            return name, path, "no_sha256", None

        try:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sha256.update(chunk)
            actual_sha = sha256.hexdigest()

            if actual_sha != expected_sha:
                return name, path, "mismatch", (expected_sha, actual_sha)
        except FileNotFoundError:
            return name, path, "missing", None
        except Exception as e:
            return name, path, "error", str(e)

        return None  # OK

    # Verify wheels concurrently
    issues = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_wheel, w): w for w in wheels}

        for future in tqdm(as_completed(futures), total=len(wheels), desc="Verifying"):
            result = future.result()
            if result:
                issues.append(result)

    if issues:
        for name, path, status, details in issues:
            console.print(f"[yellow]{name}[/yellow]:{path}")

            if status == "mismatch":
                console.print(f"  [red]Error: SHA256 mismatch![/red]")
                console.print(f"  Expected: {details[0]}")
                console.print(f"  Actual:   {details[1]}")
            elif status == "no_sha256":  # warning for missing SHA256
                console.print(f"  [yellow]Warning: No SHA256 found.[/yellow]")
            else:
                console.print(f"  [red]Status: {status}[/red]")
                if details:
                    console.print(f"  Details: {details}")

        raise typer.Exit(code=1)
    else:
        console.print(
            f"\n[green]✅ All {len(wheels)} wheels verified successfully![/green]"
        )


if __name__ == "__main__":
    app()
