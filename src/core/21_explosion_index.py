#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_explosion_index.py

Compute a simple "explosion-likeness" index (ELI) per event using existing TAMC/FRANJAMAR outputs.
Designed to be dropped into src/core and run with no CLI arguments:

    python 21_explosion_index.py

Outputs:
- resultados/_summary/explosion_index.csv
- Prints a ranked table to stdout.

Assumptions (robust to missing pieces):
- Results live under <project_root>/resultados/<event_id>/<phase>/...
- The script looks for:
    precursors/precursors_timeseries.csv
    precursors/null_summary.csv
    sync/sync_multistation_*.csv
    scan/scan_*_*.csv  (per-station extracted extremes)
If a file is missing, that component is skipped and the index is computed from available signals.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Helpers
# -------------------------
def project_root() -> Path:
    # .../src/core/21_explosion_index.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read CSV: {path} ({e})")
    return None


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer explicit relative hour columns
    for c in ["t_rel_h", "t_rel_hours", "t_hours", "t_h", "t_rel"]:
        if c in df.columns:
            return c
    # Sometimes time is stored as "t_center_h" or similar
    for c in df.columns:
        if re.search(r"(t|time).*rel.*h", c, re.IGNORECASE):
            return c
    return None


def _coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _parse_event_time_from_event_id(event_id: str) -> Optional[pd.Timestamp]:
    """
    Attempt to parse an origin timestamp from event_id strings like:
    '..._20170903_033001' or '..._20200804_150818'

    Returns UTC-naive pandas Timestamp or None.
    """
    m = re.search(r"_(\d{8})_(\d{6})(?:_|$)", event_id)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        return pd.to_datetime(ymd + hms, format="%Y%m%d%H%M%S")
    except Exception:
        return None


def _derive_event_id_and_phase(event_dir: Path) -> Tuple[str, str]:
    """
    event_dir is expected like .../resultados/<event_id>/<phase>
    """
    phase = event_dir.name
    event_id = event_dir.parent.name
    return event_id, phase


@dataclass
class EventSignals:
    event_id: str
    phase: str
    event_dir: Path
    t0: Optional[pd.Timestamp]

    # Components (None if unavailable)
    compactness: Optional[float] = None      # 0..1, higher = more concentrated near t=0
    sync_alignment: Optional[float] = None   # 0..1, higher = synchrony peak near t=0
    precursor_strength: Optional[float] = None  # 0..1, higher = stronger precursor evidence

    eli: Optional[float] = None


def _compute_compactness_from_scan(scan_dir: Path, t0: Optional[pd.Timestamp]) -> Optional[float]:
    """
    Compactness score based on per-station extracted extremes:
    fraction of high-|z| extreme points that fall within +/- 0.5 h of t=0.

    Requires a scan_*.csv family. Falls back to using any column containing 'z'
    and any time-relative column if present; otherwise attempts time_center_iso with t0.
    """
    if not scan_dir.exists():
        return None

    scan_files = sorted(scan_dir.glob("scan_*.csv"))
    if not scan_files:
        return None

    # Aggregate across stations, but limit to avoid huge memory in large datasets
    rows = []
    for f in scan_files[:400]:  # safety cap
        df = safe_read_csv(f)
        if df is None or df.empty:
            continue
        df.columns = [c.strip() for c in df.columns]
        # Find z column
        zcol = None
        for c in df.columns:
            if c.lower() in ("z", "zscore", "z_score", "z_robust", "zrobust"):
                zcol = c
                break
        if zcol is None:
            # heuristic: first column containing 'z'
            z_candidates = [c for c in df.columns if "z" in c.lower()]
            if z_candidates:
                zcol = z_candidates[0]
        if zcol is None:
            continue

        tcol = _find_time_column(df)
        if tcol:
            t = _coerce_numeric(df[tcol])
        elif "time_center_iso" in df.columns and t0 is not None:
            # compute relative hours from ISO time centers
            tt = pd.to_datetime(df["time_center_iso"], errors="coerce")
            t = (tt - t0).dt.total_seconds() / 3600.0
        else:
            continue

        z = _coerce_numeric(df[zcol]).abs()
        ok = (~t.isna()) & (~z.isna())
        if ok.sum() == 0:
            continue
        rows.append(pd.DataFrame({"t": t[ok].astype(float), "az": z[ok].astype(float)}))

    if not rows:
        return None

    all_df = pd.concat(rows, ignore_index=True)

    # Focus on "high" extremes to make this measure stable
    # Use top 10% by |z| (or at least 100 points if tiny)
    all_df = all_df.dropna()
    if all_df.empty:
        return None

    q = all_df["az"].quantile(0.90)
    hi = all_df[all_df["az"] >= q]
    if len(hi) < 50:
        # if too few, fall back to all extremes
        hi = all_df

    window = 0.5  # hours
    compact = float((hi["t"].abs() <= window).mean())
    # Normalize so that "random" uniform over 24h would be ~ (1h / 24h) ~ 0.041
    # Map to [0,1] via a gentle rescale, but keep raw in bounds.
    # rescaled = (compact - 1/24) / (1 - 1/24)
    # return max(0.0, min(1.0, rescaled))
    return max(0.0, min(1.0, compact))


def _compute_sync_alignment(sync_dir: Path, t0: Optional[pd.Timestamp]) -> Optional[float]:
    """
    Synchrony alignment score: ratio of max active fraction within +/-0.5h of t=0
    to the global max active fraction.

    Reads sync_multistation*.csv.
    """
    if not sync_dir.exists():
        return None
    files = sorted(sync_dir.glob("sync_multistation*.csv"))
    if not files:
        return None

    # Use the first one (should be unique per event)
    df = safe_read_csv(files[0])
    if df is None or df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]

    # active fraction column
    fcol = None
    for c in ["active_frac", "f_active", "f", "frac_active", "fraction_active"]:
        if c in df.columns:
            fcol = c
            break
    if fcol is None:
        # heuristic
        cand = [c for c in df.columns if "frac" in c.lower()]
        if cand:
            fcol = cand[0]
    if fcol is None:
        return None

    # time column
    tcol = _find_time_column(df)
    if tcol:
        t = _coerce_numeric(df[tcol])
    elif "time_center_iso" in df.columns and t0 is not None:
        tt = pd.to_datetime(df["time_center_iso"], errors="coerce")
        t = (tt - t0).dt.total_seconds() / 3600.0
    else:
        return None

    f = _coerce_numeric(df[fcol])
    ok = (~t.isna()) & (~f.isna())
    if ok.sum() == 0:
        return None

    t = t[ok].astype(float)
    f = f[ok].astype(float)

    gmax = float(f.max())
    if not math.isfinite(gmax) or gmax <= 0:
        return None

    window = 0.5
    local = float(f[t.abs() <= window].max()) if (t.abs() <= window).any() else 0.0
    return max(0.0, min(1.0, local / gmax))


def _compute_precursor_strength(precursors_dir: Path) -> Optional[float]:
    """
    Precursor strength based on p-values in precursors/null_summary.csv.
    Returns a score in [0,1] where 1 means strong precursor evidence (small p-values).

    If multiple p-values exist, uses the minimum.
    """
    null_path = precursors_dir / "null_summary.csv"
    df = safe_read_csv(null_path)
    if df is None or df.empty:
        return None

    # collect any columns that look like p-values
    pcols = [c for c in df.columns if re.search(r"\bp(_value)?\b", c, re.IGNORECASE) or "pval" in c.lower()]
    if not pcols:
        # sometimes stored as 'real_p'
        pcols = [c for c in df.columns if "real" in c.lower() and "p" in c.lower()]

    pvals = []
    for c in pcols:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals = vals.dropna()
        pvals.extend(list(vals.values))

    if not pvals:
        return None

    pmin = float(min(pvals))
    # Map: pmin=0 -> 1, pmin>=0.5 -> ~0
    strength = 1.0 - min(1.0, pmin / 0.5)
    return max(0.0, min(1.0, strength))


def _combine_eli(compactness: Optional[float], sync_alignment: Optional[float], precursor_strength: Optional[float]) -> Optional[float]:
    """
    Explosion-likeness index (ELI):
      high when compactness and t=0 synchrony are high, and precursor evidence is low.

      ELI = mean(available components) of:
          compactness,
          sync_alignment,
          (1 - precursor_strength)

    Returns None if nothing is available.
    """
    comps = []
    if compactness is not None:
        comps.append(compactness)
    if sync_alignment is not None:
        comps.append(sync_alignment)
    if precursor_strength is not None:
        comps.append(1.0 - precursor_strength)

    if not comps:
        return None
    return float(sum(comps) / len(comps))


def discover_event_dirs(resultados_root: Path) -> List[Path]:
    """
    Find event phase directories by locating precursors_timeseries.csv.
    This is the most reliable anchor across the pipeline.
    """
    candidates = []
    for p in resultados_root.rglob("precursors_timeseries.csv"):
        # .../resultados/<event_id>/<phase>/precursors/precursors_timeseries.csv
        event_dir = p.parent.parent  # -> <phase>
        if event_dir.name.lower() in {"precursors", "sync", "scan"}:
            continue
        candidates.append(event_dir)
    # De-dup
    uniq = []
    seen = set()
    for d in candidates:
        k = str(d.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return sorted(uniq)


def main() -> int:
    root = project_root()
    resultados = root / "resultados"
    if not resultados.exists():
        print(f"[ERROR] resultados/ not found at: {resultados}")
        print("Run the pipeline first, or adjust paths.")
        return 2

    event_dirs = discover_event_dirs(resultados)
    if not event_dirs:
        print("[ERROR] No events found (could not locate any precursors_timeseries.csv under resultados/).")
        return 2

    signals: List[EventSignals] = []
    for ev_dir in event_dirs:
        event_id, phase = _derive_event_id_and_phase(ev_dir)
        # ONLY real events (ignore control days)
        if phase.lower() != "mainshock":
            continue

        t0 = _parse_event_time_from_event_id(event_id)

        scan_dir = ev_dir / "scan"
        sync_dir = ev_dir / "sync"
        prec_dir = ev_dir / "precursors"

        comp = _compute_compactness_from_scan(scan_dir, t0)
        sync = _compute_sync_alignment(sync_dir, t0)
        prec = _compute_precursor_strength(prec_dir)
        eli = _combine_eli(comp, sync, prec)

        signals.append(EventSignals(
            event_id=event_id,
            phase=phase,
            event_dir=ev_dir,
            t0=t0,
            compactness=comp,
            sync_alignment=sync,
            precursor_strength=prec,
            eli=eli,
        ))

    # Build table
    rows = []
    for s in signals:
        rows.append({
            "event_id": s.event_id,
            "phase": s.phase,
            "eli": s.eli,
            "compactness_near_t0": s.compactness,
            "sync_alignment_t0": s.sync_alignment,
            "precursor_strength": s.precursor_strength,
            "event_dir": str(s.event_dir.relative_to(project_root())),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["eli", "event_id"], ascending=[False, True])

    out_dir = resultados / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "explosion_index.csv"
    df.to_csv(out_csv, index=False)

    # Pretty print
    show = df.copy()
    for c in ["eli", "compactness_near_t0", "sync_alignment_t0", "precursor_strength"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) and pd.notna(x) else "NA")

    print("\n=== Explosion-likeness index (ELI) — ranked ===")
    print(show[["eli", "compactness_near_t0", "sync_alignment_t0", "precursor_strength", "event_id", "phase"]].to_string(index=False))
    print(f"\nWrote: {out_csv.relative_to(project_root())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
