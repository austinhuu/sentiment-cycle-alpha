from __future__ import annotations

import json

from src.config import PROJECT_ROOT, load_config
from src.production.daily_snapshot import build_daily_snapshot


def main() -> None:
    config = load_config()
    run_dir, summary = build_daily_snapshot(
        PROJECT_ROOT, dict(config.get("v6_production", {}))
    )
    print(f"V6 production snapshot: {run_dir}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()