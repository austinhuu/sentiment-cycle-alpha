from __future__ import annotations

import json

from src.config import PROJECT_ROOT, load_config
from src.evaluation.v6_acceptance import build_v6_acceptance, save_v6_acceptance


def main() -> None:
    config = load_config()
    acceptance_config = dict(config.get("v6_acceptance", {}))
    table, summary = build_v6_acceptance(PROJECT_ROOT, acceptance_config)
    output_dir = PROJECT_ROOT / acceptance_config.get(
        "output_dir", "outputs/v6/acceptance"
    )
    save_v6_acceptance(table, summary, output_dir)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()