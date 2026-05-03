# src/core/_common.py
# Minimal compatibility layer for scripts that do: from _common import ...
# Adjust paths here if your project uses a different structure.

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd


def _project_root() -> Path:
    # src/core/_common.py -> parents[0]=core, [1]=src, [2]=project root
    return Path(__file__).resolve().parents[2]


def default_metrics_file(event: str) -> Path:
    """
    Returns default path to the real metrics file for a given event.
    This is a best-effort guess that matches your results tree style.
    If your pipeline uses a different filename, change it here.
    """
    root = _project_root()
    # Common locations used in your project:
    candidates = [
        root / "resultados" / event / "real_metrics.csv",
        root / "resultados" / "null_tests_13_14" / "analysis" / f"{event}_real_metrics.csv",
        root / "resultados" / "null_tests_13_14" / "analysis" / f"{event}_real_metrics.csv".replace("__", "_"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: allow script to fail with an explicit message
    return candidates[0]


def default_test_out_dir(event: str, test_name: str) -> Path:
    root = _project_root()
    return root / "resultados" / event / "nulltest" / test_name


def get_mainshock_time(event: str) -> datetime:
    """
    Extract mainshock time from the event string if it ends with YYYYMMDD_HHMMSS.
    Example: 2011_Great_Tohoku_..._20110311_054624
    """
    try:
        parts = event.split("_")
        # last two tokens should be date and time
        ymd = parts[-2]
        hms = parts[-1]
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except Exception as e:
        raise ValueError(
            f"Cannot parse mainshock time from event='{event}'. "
            "Expected suffix like ..._YYYYMMDD_HHMMSS"
        ) from e
