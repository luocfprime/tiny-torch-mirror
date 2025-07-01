import hashlib
import http.server
import itertools
import os
import re
import socketserver
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
import yaml
from rich.console import Console
from tqdm import tqdm

from tiny_torch_mirror.core.config import (
    CONFIG_PATH,
    PyTorchMirrorConfig,
    get_config,
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
):
    """Serve the mirror repo using HTTP server. (Run this on the same machine where the mirror is hosted)"""
    mirror_path = Path(path).expanduser().resolve()
    os.chdir(mirror_path)

    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:  # noqa
        print(f"Serving at http://0.0.0.0:{port}")
        print(f"To use the mirror, set the index URL to: http://localhost:{port}/")
        print(f"Example: pip install torch --index-url http://localhost:{port}/cu118")
        httpd.serve_forever()


@app.command()
def verify(
    config_path: Path = CONFIG_PATH,
):
    """Verify the integrity of the mirror repo. (Run this on the same machine where the mirror is hosted)"""
    config = load_config(config_path)

    wheels = fetch_existing_from_local_mirror_repo(
        Path(config.mirror_root), config.packages, config.cuda_versions
    )

    broken_wheels = []

    def verify_wheel(wheel_info):
        wheel_name, wheel_path, expected_sha256 = wheel_info

        # Calculate actual SHA256
        sha256_hash = hashlib.sha256()
        try:
            with open(wheel_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_sha256 = sha256_hash.hexdigest()

            if actual_sha256 != expected_sha256:
                return (
                    wheel_name,
                    wheel_path,
                    expected_sha256,
                    actual_sha256,
                    "checksum_mismatch",
                )
            return None

        except FileNotFoundError:
            return (wheel_name, wheel_path, expected_sha256, None, "file_not_found")
        except Exception as e:
            return (wheel_name, wheel_path, expected_sha256, None, f"error: {str(e)}")

    # Verify wheels concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_wheel, wheel): wheel for wheel in wheels}

        for future in tqdm(
            as_completed(futures), total=len(wheels), desc="Verifying wheels"
        ):
            result = future.result()
            if result:
                broken_wheels.append(result)

    # Report results
    if broken_wheels:
        console.print(f"\n[red]❌ Found {len(broken_wheels)} broken wheels:[/red]")
        for (
            wheel_name,
            wheel_path,
            expected_sha256,
            actual_sha256,
            error_type,
        ) in broken_wheels:
            console.print(f"\n[yellow]{wheel_name}[/yellow]")
            console.print(f"  Path: {wheel_path}")
            if error_type == "checksum_mismatch":
                console.print(f"  Expected: {expected_sha256}")
                console.print(f"  Actual:   {actual_sha256}")
            else:
                console.print(f"  [red]Error: {error_type}[/red]")

        raise typer.Exit(code=1)
    else:
        console.print(
            f"\n[green]✅ All {len(wheels)} wheels verified successfully![/green]"
        )


if __name__ == "__main__":
    app()
