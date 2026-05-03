
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[ES]
Comparación multi-evento de "fases" de actividad extrema.
Detecta segmentos temporales donde |z| >= umbral, los agrupa en fases
y compara su estructura temporal entre eventos (alineados a t0).

[EN]
Multi-event comparison of extreme-activity "phases".
Detects time segments where |z| >= threshold, groups them into phases,
and compares their temporal structure across events (aligned to t0).
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EVENT CONFIG / CONFIGURACIÓN EVENTOS
# ============================================================

"""
[ES]
Diccionario con el tiempo de origen (UTC) de cada evento.
Debe coincidir con los IDs usados en resultados/<evento>/metrics/

[EN]
Dictionary with origin time (UTC) for each event.
Must match the IDs used in resultados/<event>/metrics/
"""
EVENT_ORIGIN_UTC: Dict[str, str] = {
    "maule2010":   "2010-02-27T06:34:11Z",
    "tohoku2011":  "2011-03-11T05:46:23Z",
    "mexico2017":  "2017-09-19T18:14:39Z",
    "sumatra2004": "2004-12-26T00:58:53Z",
}


# ============================================================
# DATA STRUCTURES / ESTRUCTURAS
# ============================================================

@dataclass
class Phase:
    """
    [ES] Representa una fase (segmento continuo) de extremos.
    [EN] Represents a phase (continuous segment) of extremes.
    """
    start_h: float
    end_h: float
    n_points: int
    peak_abs_z: float
    mean_abs_z: float

    @property
    def duration_h(self) -> float:
        return float(self.end_h - self.start_h)


# ============================================================
# PATH / CSV HELPERS
# ============================================================

def guess_project_root(cwd: str) -> str:
    """
    [ES] Asume ejecución desde <repo>/src/core y sube dos niveles.
    [EN] Assumes execution from <repo>/src/core and goes up two levels.
    """
    return os.path.abspath(os.path.join(cwd, "..", ".."))


def find_metrics_csv(root: str, event: str) -> str:
    """
    [ES] Localiza el CSV de métricas del evento.
    [EN] Locate the event metrics CSV.
    """
    p = os.path.join(root, "resultados", event, "metrics", "tamc_24h_metrics_allstations.csv")
    if os.path.exists(p):
        return p

    # fallback: cualquier csv que contenga "allstations" en el nombre
    metrics_dir = os.path.join(root, "resultados", event, "metrics")
    if os.path.isdir(metrics_dir):
        for fn in os.listdir(metrics_dir):
            if fn.lower().endswith(".csv") and "allstations" in fn.lower():
                return os.path.join(metrics_dir, fn)

    raise FileNotFoundError(f"No encuentro metrics CSV para {event}. Miré: {p}")


def pick_z_col(df: pd.DataFrame) -> str:
    """
    [ES] Detecta automáticamente la columna de z-score.
    [EN] Auto-detects the z-score column.
    """
    for c in ["zscore", "z_score", "z", "Z", "zScore"]:
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna zscore/z_score. Columnas: {list(df.columns)}")


# ============================================================
# TIME HANDLING / MANEJO DEL TIEMPO (FIX)
# ============================================================

def ensure_t_hours(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """
    [ES]
    Garantiza df["t_hours"] (horas respecto al origen del evento) SIN requerir hardcodes.
    Orden de prioridad:
      1) Si existe alguna columna en horas (p.ej. time_center_h, time_h, t_hours...), úsala.
      2) Si hay time_center_iso:
         2a) Intenta inferir t0 del nombre del evento con patrón *_YYYYMMDD_HHMMSS
         2b) Si no, infiere t0 desde el propio CSV (timestamp de |z| máximo si hay zscore; si no, el primer timestamp)
         y calcula t_hours = (time_center_iso - t0) en horas.

    [EN]
    Ensures df["t_hours"] (hours relative to event origin) WITHOUT hardcoded dictionaries.
    Priority:
      1) Use an hour-like column if present (e.g., time_center_h, time_h, t_hours...).
      2) Else, if time_center_iso exists:
         2a) Infer t0 from event folder name pattern *_YYYYMMDD_HHMMSS
         2b) Else infer t0 from the CSV itself (time of max |zscore| if available; otherwise first timestamp)
         then compute hours relative to t0.
    """

    # 1) Already have hours
    hour_like = [
        "t_hours", "hours_from_event", "hours_rel", "t_rel_hours",
        "t_hour", "t_h", "horas", "hours",
        "time_center_h", "time_h", "t_center_h", "tcenter_h", "t_cent_h",
        "hours_since_event", "hours_since_origin",
    ]
    for c in hour_like:
        if c in df.columns:
            out = df.copy()
            out["t_hours"] = pd.to_numeric(out[c], errors="coerce")
            return out

    # 2) Need ISO timestamps
    if "time_center_iso" not in df.columns:
        raise ValueError(
            f"No encuentro columna temporal en horas ni 'time_center_iso'. Columnas: {list(df.columns)}"
        )

    out = df.copy()

    # Clean up strings
    s = out["time_center_iso"].astype(str).str.strip()
    s = s.str.replace("\u200b", "", regex=False)  # zero-width space
    s = s.str.replace("\ufeff", "", regex=False)  # BOM
    s = s.str.replace("Z", "+00:00", regex=False)

    # Parse ISO8601 robustly
    try:
        t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except TypeError:
        t = pd.to_datetime(s, errors="coerce", utc=True)

    if t.isna().all():
        raise ValueError("No pude parsear 'time_center_iso' a datetime (todo NaT).")

    # 2a) Infer t0 from event name: *_YYYYMMDD_HHMMSS
    def _t0_from_event_name(name: str):
        m = re.search(r"(\d{8})_(\d{6})", name)
        if not m:
            return None
        ymd, hms = m.group(1), m.group(2)
        try:
            dt = pd.to_datetime(ymd + hms, format="%Y%m%d%H%M%S", utc=True)
            return dt
        except Exception:
            return None

    t0 = _t0_from_event_name(event)

    # 2b) Fallback: infer from data
    if t0 is None:
        zcol = None
        for cand in ["zscore", "z", "z_score", "Z", "abs_z", "abs_zscore"]:
            if cand in out.columns:
                zcol = cand
                break

        if zcol is not None:
            try:
                z = pd.to_numeric(out[zcol], errors="coerce")
                idxmax = (z.abs()).idxmax()
                if pd.notna(idxmax) and idxmax in t.index:
                    t0 = t.loc[idxmax]
            except Exception:
                t0 = None

    if t0 is None:
        # safest: first valid timestamp
        t0 = t.dropna().iloc[0]

    out["t_hours"] = (t - t0).dt.total_seconds() / 3600.0
    return out

def detect_phases(df: pd.DataFrame, zcol: str, zthr: float, merge_gap_min: float, min_points: int) -> List[Phase]:
    """
    [ES]
    Detecta segmentos (fases) donde |z|>=zthr en el tiempo.
    Une puntos en el mismo segmento si el gap <= merge_gap_min (minutos).

    [EN]
    Detects segments (phases) where |z|>=zthr over time.
    Merges points into the same segment if gap <= merge_gap_min (minutes).
    """
    d = df.copy()
    d = d[np.isfinite(d["t_hours"])].copy()
    d["absz"] = np.abs(pd.to_numeric(d[zcol], errors="coerce"))
    d = d[np.isfinite(d["absz"])].copy()
    d = d.sort_values("t_hours")

    d_hi = d[d["absz"] >= zthr].copy()
    if d_hi.empty:
        return []

    times = d_hi["t_hours"].values
    absz = d_hi["absz"].values
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
        start_h = float(times[a])
        end_h = float(times[b])
        peak = float(np.max(absz[a:b + 1]))
        meanv = float(np.mean(absz[a:b + 1]))
        phases.append(Phase(start_h, end_h, n, peak, meanv))

    phases.sort(key=lambda p: p.start_h)
    return phases


def summarize_phases(phases: List[Phase]) -> Dict[str, float]:
    """
    [ES] Resumen para CSV: número de fases + info de primeras fases.
    [EN] Summary for CSV: number of phases + info of first phases.
    """
    out: Dict[str, float] = {}
    out["n_phases"] = float(len(phases))

    if len(phases) >= 1:
        p = phases[0]
        out.update({
            "phase1_start_h": p.start_h,
            "phase1_end_h": p.end_h,
            "phase1_duration_h": p.duration_h,
            "phase1_n": float(p.n_points),
            "phase1_peak_absz": p.peak_abs_z,
            "phase1_mean_absz": p.mean_abs_z
        })

    if len(phases) >= 2:
        p = phases[1]
        out.update({
            "phase2_start_h": p.start_h,
            "phase2_end_h": p.end_h,
            "phase2_duration_h": p.duration_h,
            "phase2_n": float(p.n_points),
            "phase2_peak_absz": p.peak_abs_z,
            "phase2_mean_absz": p.mean_abs_z,
            "gap_1_2_h": p.start_h - phases[0].end_h
        })

    return out


# ============================================================
# PLOT / GRÁFICA
# ============================================================

def plot_event_phases(df: pd.DataFrame, zcol: str, phases: List[Phase], event: str,
                      out_png: str, zthr: float, show: bool = False, nosave: bool = False):
    """
    [ES] Grafica |z| (solo puntos sobre umbral) y resalta fases.
    [EN] Plots |z| (only points above threshold) and highlights phases.
    """
    d = df.copy()
    d["absz"] = np.abs(pd.to_numeric(d[zcol], errors="coerce"))
    d = d[np.isfinite(d["t_hours"]) & np.isfinite(d["absz"])].copy()
    hi = d[d["absz"] >= zthr].copy()

    plt.figure(figsize=(11, 5))
    plt.scatter(hi["t_hours"], hi["absz"], s=8)
    plt.axvline(0, linestyle="--")
    plt.axhline(zthr, linestyle="--")

    plt.title(f"Fases |z|>={zthr} — {event}")
    plt.xlabel("Horas respecto al evento (t=0) / Hours since event (t=0)")
    plt.ylabel("|z|")

    ymax = float(hi["absz"].max()) if not hi.empty else (zthr + 1.0)
    for i, ph in enumerate(phases[:4], start=1):
        plt.axvspan(ph.start_h, ph.end_h, alpha=0.15)
        mid = 0.5 * (ph.start_h + ph.end_h)
        plt.text(mid, ymax * 0.95, f"P{i}\n{ph.duration_h:.2f}h\nn={ph.n_points}",
                 ha="center", va="top")

    plt.tight_layout()

    if not nosave:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)

    if show:
        plt.show()

    plt.close()


# ============================================================
# PIPELINE / PROCESO POR EVENTO
# ============================================================

def process_event(root: str, event: str, zthr: float, merge_gap_min: float, min_points: int):
    csv_path = find_metrics_csv(root, event)
    df = pd.read_csv(csv_path)
    zcol = pick_z_col(df)
    df = ensure_t_hours(df, event)
    phases = detect_phases(df, zcol, zthr=zthr, merge_gap_min=merge_gap_min, min_points=min_points)
    summ = summarize_phases(phases)

    row = {
        "event": event,
        "metrics_csv": csv_path,
        "zcol": zcol,
        "zthr": zthr,
        "merge_gap_min": merge_gap_min,
        "min_points": min_points,
        **summ
    }
    return row, (df, zcol, phases)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--root", default=None, help="project root (default: auto desde src/core)")
    ap.add_argument("--outdir", default=None, help="default: resultados/multi_phase_compare")
    ap.add_argument("--zthr", type=float, default=3.0)
    ap.add_argument("--merge_gap_min", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--show", action="store_true", help="[ES] Muestra figuras / [EN] Show plots")
    ap.add_argument("--nosave", action="store_true", help="[ES] No guarda PNG / [EN] Do not save PNG")
    args = ap.parse_args()

    root = args.root or guess_project_root(os.getcwd())
    outdir = args.outdir or os.path.join(root, "resultados", "multi_phase_compare")
    os.makedirs(outdir, exist_ok=True)

    print("== Multi-phase compare ==")
    print(f"Root: {root}")
    print(f"Out:  {outdir}")
    print(f"Events: {args.events}")
    print(f"Params: zthr={args.zthr}, merge_gap_min={args.merge_gap_min}, min_points={args.min_points}")
    print(f"Flags: show={args.show}, nosave={args.nosave}\n")

    rows = []
    for ev in args.events:
        print(f"[EVENT] {ev}")
        row, pack = process_event(root, ev, args.zthr, args.merge_gap_min, args.min_points)
        rows.append(row)

        df, zcol, phases = pack
        png = os.path.join(outdir, f"{ev}_phases.png")
        plot_event_phases(df, zcol, phases, ev, png, zthr=args.zthr, show=args.show, nosave=args.nosave)

        print(f"  -> phases={int(row.get('n_phases', 0))}  PNG={png}")
        if float(row.get("n_phases", 0)) >= 2:
            print(f"  -> gap_1_2_h={row.get('gap_1_2_h'):.3f}  "
                  f"P1_dur={row.get('phase1_duration_h'):.3f}  P2_dur={row.get('phase2_duration_h'):.3f}")
        print()

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, "multi_phase_summary.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
