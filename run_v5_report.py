from __future__ import annotations

from src.config import PROJECT_ROOT
from src.reporting.v5_report import save_v5_report


def main() -> None:
    path = save_v5_report(PROJECT_ROOT)
    print(f"V5 report saved: {path}", flush=True)


if __name__ == "__main__":
    main()