from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import write_mock_mvsec_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny mock MVSEC pair for local testing.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "examples" / "mock_mvsec")
    parser.add_argument("--name", type=str, default="mock_mvsec")
    args = parser.parse_args()

    h5_path, flow_path = write_mock_mvsec_pair(args.outdir, name=args.name)
    print(h5_path)
    print(flow_path)


if __name__ == "__main__":
    main()
