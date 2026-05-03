# 14_station_coverage_phases.py
# Para cada evento:
#  - detecta fases (igual que 13)
#  - calcula cobertura de estaciones en fase1/fase2:
#      stations_total, stations_active, fraction_active
# Crea:
#   resultados/multi_phase_compare/station_coverage_by_phase.csv
#   resultados/multi_phase_compare/station_coverage_compare.png
#
# Ejecuta desde: tamcsismico/src/core
# Ej:
#   python 14_station_coverage_phases.py maule2010 tohoku2011 mexico2017 sumatra2004

import os
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- MISMO diccionario que en 13 ----
EVENT_ORIGIN_UTC: Dict[str, str] = {
    "maule2010":   "2010-02-27T06:34:11Z",
    "tohoku2011":  "2011-03-11T05:46:23Z",
    "mexico2017":  "2017-09-19T18:14:39Z",
    "sumatra2004": "2004-12-26T00:58:53Z",
}


def guess_project_root(cwd: str) -> str:
    return os.path.abspath(os.path.join(cwd, "..", ".."))


def _to_datetime_utc(x: str) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"No pude parsear fecha ISO: {x}")
    return ts


def find_metrics_csv(root: str, event: str) -> str:
    p = os.path.join(root, "resultados", event, "metrics", "tamc_24h_metrics_allstations.csv")
    if os.path.exists(p):
        return p
    metrics_dir = os.path.join(root, "resultados", event, "metrics")
    if os.path.isdir(metrics_dir):
        for fn in os.listdir(metrics_dir):
            if fn.lower().endswith(".csv") and "allstations" in fn.lower():
                return os.path.join(metrics_dir, fn)
    raise FileNotFoundError(f"No encuentro metrics CSV para {event}. Miré: {p}")


def pick_z_col(df: pd.DataFrame) -> str:
    for c in ["zscore", "z_score", "z", "Z", "zScore"]:
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna zscore/z_score. Columnas: {list(df.columns)}")


def _infer_t0_from_event_name(event: str):
    """Infer t0 from event folder name pattern *_YYYYMMDD_HHMMSS (UTC)."""
    m = re.search(r"(\d{8})_(\d{6})", event)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        return pd.to_datetime(f"{ymd}{hms}", format="%Y%m%d%H%M%S", utc=True)
    except Exception:
        return None


def ensure_t_hours(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """
    Ensure column 't_hours' (hours relative to event origin) WITHOUT requiring EVENT_ORIGIN_UTC.

    Priority:
      1) If an hour-like column exists (t_hours, time_center_h, time_h, ...), use it.
      2) Else, if 'time_center_iso' exists, infer t0 from event name (YYYYMMDD_HHMMSS).
      3) Else, infer t0 from the CSV: timestamp of max |zscore| if present, otherwise first valid timestamp.
    """
    if "t_hours" in df.columns:
        return df

    hour_like = [
        "time_center_h", "time_h",
        "t_hours", "hours_from_event", "hours_rel", "t_rel_hours",
        "t_hour", "t_h", "horas", "hours", "hour",
    ]
    for c in hour_like:
        if c in df.columns:
            out = df.copy()
            out["t_hours"] = pd.to_numeric(out[c], errors="coerce")
            return out

    if "time_center_iso" not in df.columns:
        raise ValueError(
            f"No encuentro columna temporal en horas ni 'time_center_iso' para '{event}'. "
            f"Columnas: {list(df.columns)}"
        )

    out = df.copy()

    s = out["time_center_iso"].astype(str).str.strip()
    s = s.str.replace("\u200b", "", regex=False)
    s = s.str.replace("\ufeff", "", regex=False)
    s_norm = s.str.replace("Z", "+00:00", regex=False)

    t = pd.to_datetime(s_norm, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f%z")
    m = t.isna()
    if m.any():
        t2 = pd.to_datetime(s_norm[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S%z")
        t.loc[m] = t2

    m = t.isna()
    if m.any():
        t3 = pd.to_datetime(s[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f")
        t.loc[m] = t3

    m = t.isna()
    if m.any():
        t4 = pd.to_datetime(s[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S")
        t.loc[m] = t4

    if t.isna().any():
        bad = out.loc[t.isna(), "time_center_iso"].head(15).tolist()
        raise ValueError(f"Hay time_center_iso no parseables (ejemplos): {bad}")

    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC")
    else:
        t = t.dt.tz_localize("UTC")

    t0 = _infer_t0_from_event_name(event)

    if t0 is None:
        if "zscore" in out.columns:
            z = pd.to_numeric(out["zscore"], errors="coerce")
            if z.notna().any():
                idx = z.abs().idxmax()
                try:
                    t0 = t.loc[idx]
                except Exception:
                    t0 = None
        if t0 is None:
            t0 = t.iloc[0]

    out["t_hours"] = (t - t0).dt.total_seconds() / 3600.0
    return out


    if "time_center_iso" not in df.columns:
        raise ValueError(
            f"No encuentro columna temporal en horas ni 'time_center_iso'. Columnas: {list(df.columns)}"
        )
    if event not in EVENT_ORIGIN_UTC:
        raise ValueError(f"No tengo EVENT_ORIGIN_UTC para '{event}'.")

    out = df.copy()
    t0 = _to_datetime_utc(EVENT_ORIGIN_UTC[event])

    s = out["time_center_iso"].astype(str).str.strip()
    s = s.str.replace("\u200b", "", regex=False)
    s = s.str.replace("\ufeff", "", regex=False)
    s_norm = s.str.replace("Z", "+00:00", regex=False)

    t = pd.to_datetime(s_norm, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f%z")
    m = t.isna()
    if m.any():
        t2 = pd.to_datetime(s_norm[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S%z")
        t.loc[m] = t2

    m = t.isna()
    if m.any():
        t3 = pd.to_datetime(s[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f")
        t.loc[m] = t3

    m = t.isna()
    if m.any():
        t4 = pd.to_datetime(s[m], errors="coerce", format="%Y-%m-%dT%H:%M:%S")
        t.loc[m] = t4

    if t.isna().any():
        bad = out.loc[t.isna(), "time_center_iso"].head(15).tolist()
        raise ValueError(f"Hay time_center_iso no parseables (ejemplos): {bad}")

    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC")
    else:
        t = t.dt.tz_localize("UTC")

    out["t_hours"] = (t - t0).dt.total_seconds() / 3600.0
    return out


@dataclass
class Phase:
    start_h: float
    end_h: float

    @property
    def duration_h(self) -> float:
        return float(self.end_h - self.start_h)


def detect_phase_windows(df: pd.DataFrame, zcol: str, zthr: float, merge_gap_min: float, min_points: int) -> List[Phase]:
    d = df.copy()
    d = d[np.isfinite(d["t_hours"])].copy()
    d["absz"] = np.abs(pd.to_numeric(d[zcol], errors="coerce"))
    d = d[np.isfinite(d["absz"])].copy()
    d = d.sort_values("t_hours")

    hi = d[d["absz"] >= zthr].copy()
    if hi.empty:
        return []

    times = hi["t_hours"].values
    gap_h = merge_gap_min / 60.0

    segments = []
    seg_start = 0
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) > gap_h:
            segments.append((seg_start, i - 1))
            seg_start = i
    segments.append((seg_start, len(times) - 1))

    phases: List[Phase] = []
    for a, b in segments:
        n = int(b - a + 1)
        if n < min_points:
            continue
        phases.append(Phase(float(times[a]), float(times[b])))

    phases.sort(key=lambda p: p.start_h)
    return phases


def station_coverage_in_window(df: pd.DataFrame, zcol: str, zthr: float, start_h: float, end_h: float) -> Tuple[int, int, float]:
    """
    Total estaciones = uniques en df
    Activas = estaciones que tienen AL MENOS un punto con |z|>=zthr dentro de [start_h, end_h]
    """
    if "station_id" not in df.columns:
        raise ValueError("El CSV no tiene 'station_id' (necesario para cobertura).")

    d = df.copy()
    d["absz"] = np.abs(pd.to_numeric(d[zcol], errors="coerce"))
    d = d[np.isfinite(d["t_hours"]) & np.isfinite(d["absz"])].copy()

    total = int(d["station_id"].nunique())

    w = d[(d["t_hours"] >= start_h) & (d["t_hours"] <= end_h) & (d["absz"] >= zthr)]
    active = int(w["station_id"].nunique())

    frac = float(active / total) if total > 0 else float("nan")
    return total, active, frac


def plot_station_coverage(df_cov: pd.DataFrame, out_png: str):
    """
    Gráfico simple: por evento, barras fase1 vs fase2 de fracción de estaciones activas.
    """
    if df_cov.empty:
        return

    events = df_cov["event"].unique().tolist()
    y = np.arange(len(events))

    f1 = []
    f2 = []
    for ev in events:
        sub = df_cov[df_cov["event"] == ev]
        f1v = sub.loc[sub["phase"] == "phase1", "fraction_active"].values
        f2v = sub.loc[sub["phase"] == "phase2", "fraction_active"].values
        f1.append(float(f1v[0]) if len(f1v) else np.nan)
        f2.append(float(f2v[0]) if len(f2v) else np.nan)

    plt.figure(figsize=(10, 4 + 0.35 * len(events)))
    # dos barras “desplazadas” a mano (sin seaborn)
    for i, ev in enumerate(events):
        if np.isfinite(f1[i]):
            plt.plot([0, f1[i]], [i - 0.15, i - 0.15], linewidth=10)
        if np.isfinite(f2[i]):
            plt.plot([0, f2[i]], [i + 0.15, i + 0.15], linewidth=10, alpha=0.6)

    plt.yticks(y, events)
    plt.xlim(0, 1.0)
    plt.xlabel("Fracción de estaciones activas (|z|>=umbral)")
    plt.title("Cobertura de estaciones por fase (phase1 vs phase2)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default=None, help="default: resultados/multi_phase_compare")
    ap.add_argument("--zthr", type=float, default=3.0)
    ap.add_argument("--merge_gap_min", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=10)
    args = ap.parse_args()

    root = args.root or guess_project_root(os.getcwd())
    outdir = args.outdir or os.path.join(root, "resultados", "multi_phase_compare")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    print("== Station coverage by phase ==")
    print(f"Root: {root}")
    print(f"Out:  {outdir}")
    print(f"Events: {args.events}")
    print(f"Params: zthr={args.zthr}, merge_gap_min={args.merge_gap_min}, min_points={args.min_points}\n")

    for ev in args.events:
        csv_path = find_metrics_csv(root, ev)
        df = pd.read_csv(csv_path)
        zcol = pick_z_col(df)
        df = ensure_t_hours(df, ev)

        phases = detect_phase_windows(df, zcol, args.zthr, args.merge_gap_min, args.min_points)
        print(f"[EVENT] {ev}  phases_detected={len(phases)}  csv={csv_path}")

        if len(phases) >= 1:
            total, active, frac = station_coverage_in_window(df, zcol, args.zthr, phases[0].start_h, phases[0].end_h)
            rows.append({
                "event": ev, "phase": "phase1",
                "phase_start_h": phases[0].start_h, "phase_end_h": phases[0].end_h,
                "phase_duration_h": phases[0].duration_h,
                "stations_total": total, "stations_active": active, "fraction_active": frac,
                "metrics_csv": csv_path, "zcol": zcol
            })
            print(f"  phase1: active={active}/{total}  frac={frac:.3f}  dur={phases[0].duration_h:.3f}h")

        if len(phases) >= 2:
            total, active, frac = station_coverage_in_window(df, zcol, args.zthr, phases[1].start_h, phases[1].end_h)
            rows.append({
                "event": ev, "phase": "phase2",
                "phase_start_h": phases[1].start_h, "phase_end_h": phases[1].end_h,
                "phase_duration_h": phases[1].duration_h,
                "stations_total": total, "stations_active": active, "fraction_active": frac,
                "metrics_csv": csv_path, "zcol": zcol
            })
            print(f"  phase2: active={active}/{total}  frac={frac:.3f}  dur={phases[1].duration_h:.3f}h")

        print()

    df_cov = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, "station_coverage_by_phase.csv")
    df_cov.to_csv(out_csv, index=False)

    out_png = os.path.join(outdir, "station_coverage_compare.png")
    plot_station_coverage(df_cov, out_png)

    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] PNG: {out_png}")


if __name__ == "__main__":
    main()
