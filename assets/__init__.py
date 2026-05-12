"""Asset-Pfade fuer Hand, Sawyer und Benchmark-Objekte."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent

AR10_URDF   = _ROOT / "ar10_description"   / "urdf" / "ar10.urdf"
SAWYER_URDF = _ROOT / "sawyer_description" / "urdf" / "sawyer.urdf"


def benchmark_part_urdf(part_id: int) -> Path:
    pid = int(part_id)
    if not (1 <= pid <= 14):
        raise ValueError("part_id must be in [1, 14]")
    return _ROOT / "benchmark_parts" / f"benchmark_part_{pid}.urdf"
