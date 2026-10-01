from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.production.artifact_registry import ArtifactRegistry


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_daily_snapshot(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    project_root = Path(project_root)
    output_root = project_root / config.get("output_dir", "outputs/v6/production_snapshot")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_files = {
        "optimized_target_weights.csv": project_root / "outputs/v6/optimization/optimized_target_weights.csv",
        "optimizer_daily_diagnostics.csv": project_root / "outputs/v6/optimization/optimizer_daily_diagnostics.csv",
        "performance_metrics.json": project_root / "outputs/v6/backtest/performance_metrics.json",
        "latest_order_plan.csv": project_root / "outputs/v6/execution/latest_order_plan.csv",
        "latest_order_plan_summary.json": project_root / "outputs/v6/execution/latest_order_plan_summary.json",
    }
    copied: list[str] = []
    missing: list[str] = []
    for name, source in source_files.items():
        if source.exists():
            shutil.copy2(source, run_dir / name)
            copied.append(name)
        else:
            missing.append(name)

    optional_acceptance = project_root / "outputs/v6/acceptance/v6_acceptance_summary.json"
    if optional_acceptance.exists():
        shutil.copy2(optional_acceptance, run_dir / optional_acceptance.name)
        copied.append(optional_acceptance.name)

    acceptance = _read_json(project_root / "outputs/v6/acceptance/v6_acceptance_summary.json")
    v5_acceptance = _read_json(project_root / "outputs/v5/acceptance/v5_acceptance_summary.json")
    optimizer = _read_json(project_root / "outputs/v6/optimization/optimizer_summary.json")
    orders = _read_json(project_root / "outputs/v6/execution/latest_order_plan_summary.json")
    readiness_checks = {
        "v5_acceptance_passed": v5_acceptance.get("status") == "pass",
        "optimizer_success": float(optimizer.get("success_share", 0.0)) >= float(
            config.get("minimum_optimizer_success_share", 0.95)
        ),
        "order_plan_created": bool(orders),
        "required_snapshot_files_present": not missing,
    }
    ready = all(readiness_checks.values())
    summary = {
        "run_id": run_id,
        "mode": config.get("mode", "paper_research"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "readiness_checks": readiness_checks,
        "copied_files": copied,
        "missing_files": missing,
        "v5_status": v5_acceptance.get("status", "missing"),
        "v6_status": acceptance.get("status", "missing"),
        "latest_order_signal_date": orders.get("signal_date"),
    }
    (run_dir / "snapshot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    registry = ArtifactRegistry(project_root)
    manifest_paths = config.get(
        "manifest_paths",
        [
            "config/settings.yaml",
            "outputs/v5/acceptance/v5_acceptance_summary.json",
            "outputs/v6/optimization/optimized_target_weights.csv",
            "outputs/v6/backtest/performance_metrics.json",
            "outputs/v6/execution/latest_order_plan.csv",
            "outputs/v6/acceptance/v6_acceptance_summary.json",
        ],
    )
    manifest = registry.build_manifest(
        manifest_paths,
        metadata={"run_id": run_id, "mode": config.get("mode", "paper_research")},
    )
    registry.save(manifest, run_dir / "artifact_manifest.json")

    latest = output_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    return run_dir, summary