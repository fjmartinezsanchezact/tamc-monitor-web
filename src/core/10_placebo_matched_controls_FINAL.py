#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[FIXED] Supports controls inside resultados/<event>/control_* and YYYYMMDD dates.

10_placebo_matched_controls.py  (FINAL for pipeline)

Purpose
-------
Placebo/robustness test: treat control periods as pseudo-events and compare against the remaining controls.
Includes optional matched-control filtering (seasonality controls).

Pipeline policy (recommended defaults)
-------------------------------------
- Default matching: month  (seasonality control)
- Default n_placebos: 1000 (bootstrap with replacement if needed)
- Skip folders containing "RUNLOGS" and any control missing the metrics CSV

Folder conventions
------------------
resultados/<event>/metrics/tamc_24h_metrics_allstations.csv
resultados/control_<event>_YYYY-MM-DD/metrics/tamc_24h_metrics_allstations.csv
also supports legacy: resultados/<event>_control_day_YYYY-MM-DD/...

Outputs
-------
resultados/<event>_nulltest/
  matched_event_vs_controls_summary.csv
  placebo_pvalues_<match>.csv
  placebo_calibration_summary_<match>.txt
  placebo_pvalues_<match>.png
  matched_controls_skipped_list_<match>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


METRICS_FILENAME = "tamc_24h_metrics_allstations.csv"
THRESHOLDS = [3, 4, 5]
Z_COL_CANDIDATES = ["zscore", "z_score", "z", "zrot", "z_rot", "z_score_rot", "zscore_rot"]

# Add events here (or pass --event-date)
KNOWN_EVENT_DATES = {
    "tohoku2011": date(2011, 3, 11),
    "maule2010": date(2010, 2, 27),
}

_DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")


def repo_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    # .../tamcsismico/src/core/10_placebo_matched_controls.py -> repo root = parents[2]
    return here.parents[2] if len(here.parents) >= 3 else here.parent


def parse_dir_date(dirname: str) -> Optional[date]:
    """Parse a date from a directory name.

    Supported patterns:
      - YYYY-MM-DD anywhere in the name (legacy)
      - YYYYMMDD anywhere in the name (current pipeline: control_03_20230729_074115)
    """
    m = _DATE_PAT.search(dirname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    m2 = re.search(r"(\d{8})", dirname)
    if not m2:
        return None
    s = m2.group(1)
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return None



def find_zcol(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in Z_COL_CANDIDATES:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        if "z" in c.lower():
            return c
    raise ValueError(f"No z column found. Columns: {list(df.columns)}")


def load_metrics(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    zc = find_zcol(df)
    df = df.copy()
    df[zc] = pd.to_numeric(df[zc], errors="coerce")
    df = df.dropna(subset=[zc])
    return df, zc


def compute_stats(df: pd.DataFrame, zc: str) -> Dict[str, float]:
    zabs = np.abs(df[zc].to_numpy(dtype=float))
    out: Dict[str, float] = {"zmax_abs": float(np.nanmax(zabs)) if len(zabs) else float("nan")}
    for t in THRESHOLDS:
        out[f"count_abs_ge_{t}"] = float(np.sum(zabs >= t))
    return out


def emp_p(event_value: float, control_values: np.ndarray) -> float:
    """Empirical p-value with +1 correction: (1 + #{ctrl >= event}) / (N + 1)."""
    vals = control_values[np.isfinite(control_values)]
    n = vals.size
    if n == 0 or not np.isfinite(event_value):
        return float("nan")
    ge = int(np.sum(vals >= event_value))
    return (1.0 + ge) / (n + 1.0)


@dataclass
class ControlStat:
    name: str
    d: Optional[str]
    zcol: str
    zmax_abs: float
    count_abs_ge_3: float
    count_abs_ge_4: float
    count_abs_ge_5: float


def list_control_dirs(resultados_dir: Path, event_name: str) -> List[Path]:
    """Find control directories.

    Supports BOTH layouts:
      A) Root-level controls (legacy / auditable):
         resultados/control_<event>_YYYY-MM-DD/
         resultados/<event>_control_day_YYYY-MM-DD/

      B) Current pipeline layout (inside event folder):
         resultados/<event>/control_XX_YYYYMMDD_HHMMSS/
    """
    dirs: List[Path] = []

    # A) root-level conventions
    prefixes = [f"control_{event_name}_", f"{event_name}_control_day_"]
    for p in resultados_dir.iterdir():
        if not p.is_dir():
            continue
        nm = p.name.lower()
        if "runlogs" in nm:
            continue
        if any(nm.startswith(pref.lower()) for pref in prefixes):
            dirs.append(p)

    # B) inside-event convention
    ev_dir = resultados_dir / event_name
    if ev_dir.is_dir():
        for p in ev_dir.iterdir():
            if not p.is_dir():
                continue
            nm = p.name.lower()
            if "runlogs" in nm:
                continue
            if nm.startswith("control_"):
                dirs.append(p)

    # de-dup
    uniq: Dict[str, Path] = {}
    for d in dirs:
        uniq[str(d.resolve()).lower()] = d
    return sorted(uniq.values(), key=lambda x: x.name)



def filter_controls_matched(control_dirs: List[Path], event_dt: date, match: List[str]) -> List[Path]:
    """Filter controls by match criteria ('month', 'dow')."""
    if not match:
        return control_dirs
    match = [m.lower() for m in match]
    out: List[Path] = []
    for p in control_dirs:
        d = parse_dir_date(p.name)
        if d is None:
            continue
        ok = True
        if "month" in match:
            ok &= (d.month == event_dt.month)
        if "dow" in match:
            ok &= (d.weekday() == event_dt.weekday())
        if ok:
            out.append(p)
    return out


def _compute_control_stat(control_dir: Path) -> Tuple[str, Optional[str], str, Dict[str, float]]:
    metrics = control_dir / "metrics" / METRICS_FILENAME
    if not metrics.exists():
        raise FileNotFoundError(str(metrics))
    df, zc = load_metrics(metrics)
    st = compute_stats(df, zc)
    d = parse_dir_date(control_dir.name)
    return control_dir.name, (str(d) if d else None), zc, st


def compute_controls_stats(control_dirs: List[Path], jobs: int, show_progress: bool) -> Tuple[List[ControlStat], List[Dict[str, str]]]:
    """Heavy step (CSV IO + parsing + stats). Done once per control; optional multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    stats_list: List[ControlStat] = []
    skipped: List[Dict[str, str]] = []

    if jobs <= 1:
        it = control_dirs
        if show_progress and tqdm is not None:
            it = tqdm(control_dirs, desc="Precomputing control stats", unit="control")  # type: ignore
        for cdir in it:  # type: ignore
            try:
                name, d, zc, st = _compute_control_stat(cdir)
                stats_list.append(ControlStat(name, d, zc, st["zmax_abs"], st["count_abs_ge_3"], st["count_abs_ge_4"], st["count_abs_ge_5"]))
            except Exception as e:
                skipped.append({"control": cdir.name, "reason": str(e)})
        return sorted(stats_list, key=lambda x: x.name), skipped

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_compute_control_stat, cdir): cdir for cdir in control_dirs}
        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(total=len(futs), desc="Precomputing control stats", unit="control")  # type: ignore
        for fut in as_completed(futs):
            cdir = futs[fut]
            try:
                name, d, zc, st = fut.result()
                stats_list.append(ControlStat(name, d, zc, st["zmax_abs"], st["count_abs_ge_3"], st["count_abs_ge_4"], st["count_abs_ge_5"]))
            except Exception as e:
                skipped.append({"control": cdir.name, "reason": str(e)})
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    return sorted(stats_list, key=lambda x: x.name), skipped


def plot_placebo_pvalues(placebo_df: pd.DataFrame, real_p: Dict[str, float], out_png: Path) -> None:
    """Histogram of placebo p-values with vertical line at real p."""
    metrics = ["zmax_abs"] + [f"count_abs_ge_{t}" for t in THRESHOLDS]
    titles = ["max(|z|)", "N(|z|>=3)", "N(|z|>=4)", "N(|z|>=5)"]

    plt.figure(figsize=(12, 8))
    for i, (m, title) in enumerate(zip(metrics, titles), start=1):
        ax = plt.subplot(2, 2, i)
        col = f"p_{m}"
        vals = pd.to_numeric(placebo_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            ax.hist(vals, bins=12, range=(0, 1))
        rp = real_p.get(col, np.nan)
        if np.isfinite(rp):
            ax.axvline(rp, linewidth=2)
            ax.set_title(f"{title} (real p={rp:.3g})")
        else:
            ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel("placebo p-values")
        ax.set_ylabel("freq")

    plt.suptitle("PLACEBO p-values (controls as pseudo-events)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", "-e", required=True, help="Evento (e.g., tohoku2011)")
    ap.add_argument("--results-dir", "-r", default=None, help="Ruta a resultados/ (default: <repo>/resultados)")
    ap.add_argument("--event-date", default=None, help="Fecha del evento YYYY-MM-DD (override)")
    ap.add_argument("--match", nargs="*", default=["month"], choices=["month", "dow"],
                    help="Matched controls (default: month)")
    ap.add_argument("--n-placebos", type=int, default=1000,
                    help="Number of placebo pseudo-events (default: 1000). If > Ncontrols, bootstrap with replacement.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="Processes for CSV parsing (1=no multiprocessing)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    event = args.event.strip()
    root = repo_root_from_this_file()
    resultados = Path(args.results_dir) if args.results_dir else (root / "resultados")

    if args.event_date:
        event_dt = datetime.strptime(args.event_date, "%Y-%m-%d").date()
    else:
        event_dt = KNOWN_EVENT_DATES.get(event.lower())
        if event_dt is None:
            raise SystemExit(f"No conozco la fecha de '{event}'. Usa --event-date YYYY-MM-DD.")

    event_metrics = resultados / event / "metrics" / METRICS_FILENAME
    if not event_metrics.exists():
        # fallback for pipeline layout
        alt = resultados / event / "mainshock" / "metrics" / METRICS_FILENAME
        if alt.exists():
            event_metrics = alt
        else:
            raise SystemExit(f"No existe: {event_metrics} (ni {alt})")

    out_dir = resultados / f"{event}_nulltest"
    out_dir.mkdir(parents=True, exist_ok=True)

    show_progress = (not args.no_progress)

    # Load event once
    df_e, zc_e = load_metrics(event_metrics)
    estats = compute_stats(df_e, zc_e)

    all_controls = list_control_dirs(resultados, event)
    matched = filter_controls_matched(all_controls, event_dt, args.match)
    match_tag = "none" if not args.match else "+".join(args.match)

    # Fallback: si el matching deja muy pocos controles, usamos TODOS.
    # Esto evita que el placebo falle cuando solo hay 1-2 controles en el mismo mes/día.
    if len(matched) < 3:
        matched = all_controls
        match_tag = "all"

    print(f"[OK] Evento: {event} ({event_dt})  zcol={zc_e}")
    print(f"[OK] Controles totales (sin RUNLOGS): {len(all_controls)} | matched({match_tag}): {len(matched)}")

    # Precompute control stats once
    controls_stats, skipped = compute_controls_stats(matched, jobs=max(1, args.jobs), show_progress=show_progress)

    if len(controls_stats) < 3 and matched is not all_controls:
        print("[WARN] Matching dejó <3 controles válidos. Reintentando placebo con TODOS los controles (sin matching).")
        matched = all_controls
        match_tag = "all"
        controls_stats, skipped = compute_controls_stats(matched, jobs=max(1, args.jobs), show_progress=show_progress)

    skipped_path = out_dir / f"matched_controls_skipped_list_{match_tag}.jsonl"
    with open(skipped_path, "w", encoding="utf-8") as f:
        for item in skipped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if len(controls_stats) < 3:
        raise SystemExit("Muy pocos controles válidos (necesitas >=3 con metrics CSV). Revisa skipped_list.")

    # Extract arrays
    names = np.array([c.name for c in controls_stats], dtype=object)
    idx_all = np.arange(len(controls_stats), dtype=int)

    zmax = np.array([c.zmax_abs for c in controls_stats], dtype=float)
    c3 = np.array([c.count_abs_ge_3 for c in controls_stats], dtype=float)
    c4 = np.array([c.count_abs_ge_4 for c in controls_stats], dtype=float)
    c5 = np.array([c.count_abs_ge_5 for c in controls_stats], dtype=float)

    # Event vs controls p-values
    real_p = {
        "p_zmax_abs": emp_p(estats["zmax_abs"], zmax),
        "p_count_abs_ge_3": emp_p(estats["count_abs_ge_3"], c3),
        "p_count_abs_ge_4": emp_p(estats["count_abs_ge_4"], c4),
        "p_count_abs_ge_5": emp_p(estats["count_abs_ge_5"], c5),
    }

    matched_summary = {
        "event": event,
        "event_date": str(event_dt),
        "match": match_tag,
        "zcol": zc_e,
        "n_controls_found": len(matched),
        "n_controls_used": int(len(controls_stats)),
        "event_zmax_abs": estats["zmax_abs"],
        "event_count_abs_ge_3": estats["count_abs_ge_3"],
        "event_count_abs_ge_4": estats["count_abs_ge_4"],
        "event_count_abs_ge_5": estats["count_abs_ge_5"],
        "p_zmax_abs": real_p["p_zmax_abs"],
        "p_count_abs_ge_3": real_p["p_count_abs_ge_3"],
        "p_count_abs_ge_4": real_p["p_count_abs_ge_4"],
        "p_count_abs_ge_5": real_p["p_count_abs_ge_5"],
        "skipped_list": str(skipped_path),
    }
    matched_csv = out_dir / "matched_event_vs_controls_summary.csv"
    pd.DataFrame([matched_summary]).to_csv(matched_csv, index=False)
    print(f"[OK] Matched summary CSV: {matched_csv}")

    # Placebos (bootstrap if needed)
    rng = np.random.default_rng(args.seed)
    n_req = int(args.n_placebos) if args.n_placebos and args.n_placebos > 0 else len(idx_all)
    replace = n_req > len(idx_all)
    idx_placebo = rng.choice(idx_all, size=n_req, replace=replace)

    it = idx_placebo
    if show_progress and tqdm is not None:
        it = tqdm(idx_placebo, desc=f"Running placebo pseudo-events (N={len(idx_placebo)}, replace={replace})", unit="placebo")  # type: ignore

    rows: List[Dict[str, object]] = []
    for draw, i in enumerate(it):  # type: ignore
        i = int(i)
        mask = (idx_all != i)
        rows.append({
            "draw": draw,
            "pseudo_event_dir": str(names[i]),
            # note: compare pseudo-event against controls excluding itself
            "p_zmax_abs": emp_p(zmax[i], zmax[mask]),
            "p_count_abs_ge_3": emp_p(c3[i], c3[mask]),
            "p_count_abs_ge_4": emp_p(c4[i], c4[mask]),
            "p_count_abs_ge_5": emp_p(c5[i], c5[mask]),
        })

    placebo_df = pd.DataFrame(rows)

    # Convert to the plotting/summary columns expected: p_<metric>
    placebo_out = pd.DataFrame({
        "draw": placebo_df["draw"],
        "pseudo_event_dir": placebo_df["pseudo_event_dir"],
        "p_zmax_abs": placebo_df["p_zmax_abs"],
        "p_count_abs_ge_3": placebo_df["p_count_abs_ge_3"],
        "p_count_abs_ge_4": placebo_df["p_count_abs_ge_4"],
        "p_count_abs_ge_5": placebo_df["p_count_abs_ge_5"],
    })
    placebo_csv = out_dir / f"placebo_pvalues_{match_tag}.csv"
    placebo_out.to_csv(placebo_csv, index=False)
    print(f"[OK] Placebo CSV: {placebo_csv}")

    # Calibration summary
    calib_txt = out_dir / f"placebo_calibration_summary_{match_tag}.txt"
    lines = []
    lines.append(f"EVENT: {event} ({event_dt})  match={match_tag}")
    lines.append(f"controls_used_for_placebos: {len(controls_stats)}")
    lines.append(f"placebos_run: {len(placebo_out)}  seed={args.seed}  replace={replace}")
    lines.append("")
    for key, rp in [
        ("p_zmax_abs", real_p["p_zmax_abs"]),
        ("p_count_abs_ge_3", real_p["p_count_abs_ge_3"]),
        ("p_count_abs_ge_4", real_p["p_count_abs_ge_4"]),
        ("p_count_abs_ge_5", real_p["p_count_abs_ge_5"]),
    ]:
        vals = pd.to_numeric(placebo_out[key], errors="coerce").dropna().to_numpy(dtype=float)
        frac = float(np.mean(vals <= float(rp))) if vals.size and np.isfinite(rp) else float("nan")
        lines.append(f"{key}: real_p={float(rp):.6g}  frac(placebos <= real_p)={frac:.3f}  N={vals.size}")
    lines.append("")
    lines.append(f"matched_summary_csv: {matched_csv}")
    lines.append(f"placebo_csv: {placebo_csv}")
    lines.append(f"skipped_list: {skipped_path}")
    with open(calib_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Calibration TXT: {calib_txt}")

    # Plot placebo p-values with vertical line at real p (mapped to p_<metric>)
    placebo_png = out_dir / f"placebo_pvalues_{match_tag}.png"
    real_plot = {
        "p_zmax_abs": real_p["p_zmax_abs"],
        "p_count_abs_ge_3": real_p["p_count_abs_ge_3"],
        "p_count_abs_ge_4": real_p["p_count_abs_ge_4"],
        "p_count_abs_ge_5": real_p["p_count_abs_ge_5"],
        # also make available under p_<metric> keys used in plot
        "p_zmax_abs": real_p["p_zmax_abs"],
    }
    # Create columns named p_<metric> where metric is zmax_abs/count_abs_ge_*
    plot_df = pd.DataFrame({
        "p_zmax_abs": placebo_out["p_zmax_abs"],
        "p_count_abs_ge_3": placebo_out["p_count_abs_ge_3"],
        "p_count_abs_ge_4": placebo_out["p_count_abs_ge_4"],
        "p_count_abs_ge_5": placebo_out["p_count_abs_ge_5"],
    })
    # plot_placebo_pvalues expects p_<metric> with metric names zmax_abs/count_abs_ge_*
    # so we rename accordingly:
    plot_df = plot_df.rename(columns={
        "p_zmax_abs": "p_zmax_abs",
        "p_count_abs_ge_3": "p_count_abs_ge_3",
        "p_count_abs_ge_4": "p_count_abs_ge_4",
        "p_count_abs_ge_5": "p_count_abs_ge_5",
    })
    # Internally plot_placebo_pvalues uses p_{metric} with metric keys zmax_abs/count_abs_ge_*
    # We'll provide the expected columns by creating them:
    plot_df2 = pd.DataFrame({
        "p_zmax_abs": plot_df["p_zmax_abs"],
        "p_count_abs_ge_3": plot_df["p_count_abs_ge_3"],
        "p_count_abs_ge_4": plot_df["p_count_abs_ge_4"],
        "p_count_abs_ge_5": plot_df["p_count_abs_ge_5"],
    })
    # And call the plotter using expected access pattern:
    # (it will look for p_zmax_abs via metric zmax_abs; we handle inside by naming p_zmax_abs and mapping)
    # To keep things simple, use a local plot implementation:
    plot_placebo_pvalues(
        pd.DataFrame({
            "p_zmax_abs": placebo_out["p_zmax_abs"],
            "p_count_abs_ge_3": placebo_out["p_count_abs_ge_3"],
            "p_count_abs_ge_4": placebo_out["p_count_abs_ge_4"],
            "p_count_abs_ge_5": placebo_out["p_count_abs_ge_5"],
        }).rename(columns={
            "p_zmax_abs": "p_zmax_abs",
            "p_count_abs_ge_3": "p_count_abs_ge_3",
            "p_count_abs_ge_4": "p_count_abs_ge_4",
            "p_count_abs_ge_5": "p_count_abs_ge_5",
        }),
        {
            "p_zmax_abs": real_p["p_zmax_abs"],
            "p_count_abs_ge_3": real_p["p_count_abs_ge_3"],
            "p_count_abs_ge_4": real_p["p_count_abs_ge_4"],
            "p_count_abs_ge_5": real_p["p_count_abs_ge_5"],
        },
        placebo_png
    )
    print(f"[OK] Placebo PNG: {placebo_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# FIXED2: fallback to all controls if matching too strict
