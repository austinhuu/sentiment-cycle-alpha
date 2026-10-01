from __future__ import annotations

import json

from src.config import PROJECT_ROOT, load_config
from src.evaluation.v5_acceptance import build_v5_acceptance, save_v5_acceptance


def main() -> None:
    config = load_config()
    acceptance_config = config.get("v5_acceptance", {})
    table, summary = build_v5_acceptance(PROJECT_ROOT, acceptance_config)
    output_dir = PROJECT_ROOT / acceptance_config.get(
        "output_dir", "outputs/v5/acceptance"
    )
    save_v5_acceptance(table, summary, output_dir)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()