from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse


def create_app(
    mirror_root: Path,
    fallback_index: str,
    path_mappings: Optional[Dict[str, str]] = None,
) -> FastAPI:
    """Create FastAPI app with configuration."""
    app = FastAPI(title="PyTorch Mirror Server")

    path_mappings = path_mappings or {}
    client = httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": "pip/22.0.4"}
    )

    def transform_path_for_fallback(path: str) -> str:
        """Transform local path to fallback path based on mappings."""
        path = path.lstrip("/")

        for local_prefix, remote_prefix in path_mappings.items():
            local_prefix = local_prefix.strip("/")
            if path.startswith(local_prefix + "/") or path == local_prefix:
                remainder = path[len(local_prefix) :].lstrip("/")
                if remote_prefix:
                    return remote_prefix.strip("/") + "/" + remainder
                else:
                    return remainder

        return path

    @app.get("/{path:path}")
    async def serve_file(path: str):
        local_path = mirror_root / path

        if local_path.exists():
            if local_path.is_file():
                return FileResponse(local_path)
            elif local_path.is_dir() and (local_path / "index.html").exists():
                return FileResponse(local_path / "index.html")

        # Redirect to fallback for missing content
        path_parts = path.rstrip("/").split("/")
        package_name = path_parts[-1]

        if package_name and not path.endswith(".whl"):
            # Redirect package index requests
            redirect_url = f"{fallback_index.rstrip('/')}/{package_name}/"
            return RedirectResponse(url=redirect_url, status_code=302)
        else:
            # Let pip handle the 404
            raise HTTPException(status_code=404, detail="Not found")

    return app
