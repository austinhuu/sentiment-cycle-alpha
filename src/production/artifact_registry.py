from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    exists: bool
    size_bytes: int
    sha256: str
    modified_utc: str


class ArtifactRegistry:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def inspect(self, relative_path: str | Path) -> ArtifactRecord:
        relative = Path(relative_path)
        path = relative if relative.is_absolute() else self.project_root / relative
        if not path.exists() or not path.is_file():
            return ArtifactRecord(str(relative).replace("\\", "/"), False, 0, "", "")
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        return ArtifactRecord(
            str(path.relative_to(self.project_root)).replace("\\", "/"),
            True,
            int(stat.st_size),
            self._sha256(path),
            modified,
        )

    def build_manifest(
        self,
        paths: Iterable[str | Path],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = [asdict(self.inspect(path)) for path in paths]
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "metadata": metadata or {},
            "artifacts": records,
            "all_required_exist": all(record["exists"] for record in records),
        }

    @staticmethod
    def save(manifest: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")