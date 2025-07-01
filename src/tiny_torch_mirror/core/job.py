import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import httpx
import stamina
from tqdm import tqdm

from tiny_torch_mirror.core.config import get_config
from tiny_torch_mirror.core.log import logger
from tiny_torch_mirror.core.sync import rsync
from tiny_torch_mirror.core.utils import (
    get_wheel_dest_path,
    parse_index_page,
    remote_cat,
    remote_ls,
    remote_mkdir,
    remote_update_index_html,
)

# a lock to ensure only one thread can access the index.html file at a time
index_lock = threading.Lock()


@stamina.retry(
    on=Exception,
    attempts=3,
    timeout=None,
    wait_initial=1.0,
    wait_max=10.0,
    wait_jitter=True,
)
def single_job(wheel_name: str, download_url: str, sha256: str):
    config = get_config()
    local_cache_dir = Path(config.local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    local_file_path = local_cache_dir / wheel_name

    # e.g. https://download.pytorch.org/whl/cu117/xxx
    cuda_version = re.search(r"cu\d+", download_url).group(0)

    dest_file_path = get_wheel_dest_path(wheel_name, cuda_version)
    dest_rel_url = "/" + str(dest_file_path.relative_to(config.mirror_root))

    # 1. Download the wheel file to local cache
    with httpx.stream("GET", download_url) as response:
        response.raise_for_status()
        with open(local_file_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    # Verify checksum
    if sha256:
        calculated_sha256 = hashlib.sha256()
        with open(local_file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                calculated_sha256.update(chunk)

        if calculated_sha256.hexdigest() != sha256:
            # Clean up the corrupted file
            local_file_path.unlink()
            raise ValueError(
                f"SHA256 mismatch for {wheel_name}:\n"
                f"  Expected: {sha256}\n"
                f"  Got:      {calculated_sha256.hexdigest()}"
            )

    # 2. Upload the wheel to remote mirror repo
    remote_mkdir(dest_file_path.parent)
    rsync(local_file_path, dest_file_path)

    # 3. Update the index.html file
    with index_lock:
        html_path = dest_file_path.parent / "index.html"

        if "index.html" in remote_ls(html_path.parent):  # index.html exists
            html_content = remote_cat(html_path)
            wheels = parse_index_page(html_content, "/")
        else:
            wheels = []

        remote_update_index_html(
            html_path, wheels=wheels + [(wheel_name, dest_rel_url, sha256)]
        )

    # 4. Clean up the local cache
    local_file_path.unlink()


def run_jobs_in_threadpool(jobs: List[Tuple[str, str, str]]):
    """
    Run multiple download jobs in a thread pool.
    :param jobs: List of tuples containing (wheel_name, download_url, sha256).
    """
    config = get_config()
    max_workers = getattr(config, "max_workers", 5)

    failed_jobs = []
    successful_jobs = []

    pbar = tqdm(total=len(jobs), desc="Downloading wheels", unit="wheel")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(single_job, wheel_name, download_url, sha256): (
                wheel_name,
                download_url,
                sha256,
            )
            for wheel_name, download_url, sha256 in jobs
        }

        for future in as_completed(futures):
            wheel_name, download_url, sha256 = futures[future]

            try:
                future.result()
                successful_jobs.append(wheel_name)
                pbar.set_postfix_str(f"✓ {wheel_name}")
            except Exception as e:
                failed_jobs.append((wheel_name, str(e)))
                pbar.set_postfix_str(f"✗ {wheel_name}")
                logger.error(f"Failed to download {wheel_name}")

            pbar.update(1)

    pbar.close()

    print(f"\n[bold]Download Summary:[/bold]")
    print(f"  [green]✓ Successful:[/green] {len(successful_jobs)}")
    print(f"  [red]✗ Failed:[/red] {len(failed_jobs)}")

    if failed_jobs:
        print(f"\n[red]Failed downloads:[/red]")
        for wheel_name, error in failed_jobs[:10]:
            print(f"  • {wheel_name}")
            print(f"    Error: {error}")
        if len(failed_jobs) > 10:
            print(f"  ... and {len(failed_jobs) - 10} more")
