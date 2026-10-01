from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"[V6 DAILY] {script}", flush=True)
    completed = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    # Research-paper daily batch. It creates targets and orders but never sends them.
    for script in [
        "run_v6_optimize.py",
        "run_v6_order_plan.py",
        "run_v6_snapshot.py",
        "run_v6_acceptance.py",
        "run_v6_snapshot.py",
        "run_v6_report.py",
    ]:
        run(script)


if __name__ == "__main__":
    main()