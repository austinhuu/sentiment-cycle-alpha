from __future__ import annotations

from src.config import PROJECT_ROOT
from src.reporting.v6_report import build_v6_report, save_v6_report


def main() -> None:
    report = build_v6_report(PROJECT_ROOT)
    path = PROJECT_ROOT / "outputs" / "v6" / "V6_FINAL_REPORT.md"
    save_v6_report(report, path)
    print(report, flush=True)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()