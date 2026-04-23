from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.smoke import run_smoke


def main() -> None:
    results = run_smoke()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
