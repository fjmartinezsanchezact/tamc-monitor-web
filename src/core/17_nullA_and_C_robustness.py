#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[ES] 17_nullA_and_C_robustness.py (FAST)
[EN] 17_nullA_and_C_robustness.py (FAST)

[ES] Optimizado para velocidad:
 - NO usa pandas dentro del Monte Carlo
 - Precalcula arrays numpy: tiempos, estaciones codificadas, |z|
 - NullA: muestrea tiempos para puntos activos (sin reemplazo)
 - RobustC: submuestreo de estaciones con máscara numpy
 - Backend: threads o processes (Windows-friendly)

[EN] Speed-optimized:
 - No pandas inside Monte Carlo loop
 - Precompute numpy arrays: times, station codes, |z|
 - NullA: sample times for active points (no replacement)
 - RobustC: station subsampling via numpy mask
 - Backend: threads or processes
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# [ES] Evitar GUI/Tkinter (Windows)
# [EN] Avoid GUI/Tkinter (Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# ============================================================
# [ES] Paths / [EN] Paths
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_event_metrics_csv(root: Path, event: str) -> Path:
    """
    [ES] Encuentra el CSV del evento en el layout TAMC:
         resultados/<evento>/metrics/tamc_24h_metrics_allstations.csv
    [EN] Finds event CSV in TAMC layout.
    """
    p = root / "resultados" / event / "metrics" / "tamc_24h_metrics_allstations.csv"
    if p.exists():
        return p

    base = root / "resultados" / event
    if base.exists():
        cands = list(base.glob("**/*tamc_24h_metrics*allstations*.csv"))
        if cands:
            return cands[0]
        cands = list(base.glob("**/*tamc_24h_metrics*.csv"))
        if cands:
            for c in cands:
                if "allstations" in c.name.lower():
                    return c
            return cands[0]

    raise FileNotFoundError(f"[ERROR] No encuentro CSV de métricas para {event}. Esperaba: {p}")


# ============================================================
# [ES] Column autodetect / [EN] Column autodetect
# ============================================================

def pick_col(cands: List[str], cols: List[str]) -> Optional[str]:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def autodetect_columns(df: pd.DataFrame, timecol: str, stacol: str, zcol: str) -> Tuple[str, str, str]:
    """
    [ES] Autodetecta nombres reales en tu CSV (station_id, zscore, etc.)
    [EN] Autodetects real column names in your CSV (station_id, zscore, etc.)
    """
    cols = list(df.columns)

    if timecol not in cols:
        tc = pick_col(["time_center_iso", "time_iso", "time", "t", "time_center"], cols)
        if tc:
            timecol = tc

    if stacol not in cols:
        sc = pick_col(["station_id", "station", "sta", "stationcode", "station_code"], cols)
        if sc:
            stacol = sc

    if zcol not in cols:
        zc = pick_col(["zscore", "z", "z_score", "tamc_z", "tamc_zscore", "tamc_rot"], cols)
        if zc:
            zcol = zc

    need = {timecol, stacol, zcol}
    if not need.issubset(set(cols)):
        raise KeyError(f"[ERROR] CSV no tiene columnas {need}. Tiene: {cols}")

    return timecol, stacol, zcol


# ============================================================
# [ES] Phase logic (numpy) / [EN] Phase logic (numpy)
# ============================================================

@dataclass
class Phase:
    start_h: float
    end_h: float
    n_points: int

    @property
    def dur_h(self) -> float:
        return float(self.end_h - self.start_h)


def parse_time_to_hours(time_series: pd.Series) -> np.ndarray:
    """
    [ES] Convierte timestamps a horas relativas (t=0 en max timestamp).
    [EN] Convert timestamps to relative hours (t=0 at max timestamp).
    """
    if np.issubdtype(time_series.dtype, np.number):
        return time_series.to_numpy(dtype=float)

    t = pd.to_datetime(time_series, utc=True, errors="coerce", format="mixed")
    if t.isna().all():
        raise ValueError("[ERROR] No puedo parsear timecol a datetime.")
    t0 = t.max()
    return ((t - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0)


def detect_phases_from_active_times(t_act: np.ndarray, merge_gap_min: float, min_points: int) -> List[Phase]:
    """
    [ES] Detecta fases usando SOLO los tiempos activos (|z|>=zthr).
    [EN] Detect phases using ONLY active times (|z|>=zthr).
    """
    if t_act.size == 0:
        return []

    t = np.sort(t_act.astype(float))
    gap_h = float(merge_gap_min) / 60.0
    dt = np.diff(t)

    cuts = np.where(dt > gap_h)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, t.size - 1]

    phases: List[Phase] = []
    for s_i, e_i in zip(starts, ends):
        npts = int(e_i - s_i + 1)
        if npts < int(min_points):
            continue
        phases.append(Phase(start_h=float(t[s_i]), end_h=float(t[e_i]), n_points=npts))

    return phases


def compute_metrics_from_active(t_act: np.ndarray,
                                sta_act: np.ndarray,
                                n_stations_total: int,
                                merge_gap_min: float,
                                min_points: int) -> Dict[str, float]:
    """
    [ES] Calcula métricas a partir de (t_act, sta_act) ya filtrados por actividad.
    [EN] Compute metrics from (t_act, sta_act) already active-filtered.
    """
    phases = detect_phases_from_active_times(t_act, merge_gap_min, min_points)

    out: Dict[str, float] = {}
    out["n_phases"] = float(len(phases))
    out["phase1_duration_h"] = phases[0].dur_h if len(phases) >= 1 else np.nan
    out["phase2_duration_h"] = phases[1].dur_h if len(phases) >= 2 else np.nan
    out["gap_1_2_h"] = (phases[1].start_h - phases[0].end_h) if len(phases) >= 2 else np.nan

    def frac_active_in_phase(ph: Optional[Phase]) -> float:
        if ph is None or n_stations_total <= 0:
            return np.nan
        m = (t_act >= ph.start_h) & (t_act <= ph.end_h)
        if not np.any(m):
            return 0.0
        return float(np.unique(sta_act[m]).size) / float(n_stations_total)

    out["phase1_fraction_active"] = frac_active_in_phase(phases[0] if len(phases) >= 1 else None)
    out["phase2_fraction_active"] = frac_active_in_phase(phases[1] if len(phases) >= 2 else None)

    return out


# ============================================================
# [ES] Monte Carlo kernels (FAST) / [EN] Monte Carlo kernels (FAST)
# ============================================================

def sim_nullA(seed: int,
              t_all: np.ndarray,
              sta_act: np.ndarray,
              n_stations_total: int,
              n_active: int,
              merge_gap_min: float,
              min_points: int) -> Dict[str, float]:
    """
    [ES] NullA FAST:
      - En un shuffle completo, los puntos activos reciben tiempos aleatorios del conjunto total.
      - Equivalente a muestrear SIN reemplazo n_active tiempos de t_all y emparejar aleatorio con sta_act.
    [EN] NullA FAST:
      - Active points get random times from all times.
      - Equivalent to sampling without replacement from t_all (size=n_active) and randomly pairing with sta_act.
    """
    rng = np.random.default_rng(int(seed))
    t_samp = rng.choice(t_all, size=n_active, replace=False)
    rng.shuffle(t_samp)  # random pairing with stations
    return compute_metrics_from_active(t_samp, sta_act, n_stations_total, merge_gap_min, min_points)


def sim_robC(seed: int,
             t_act: np.ndarray,
             sta_act: np.ndarray,
             unique_stas: np.ndarray,
             keep_frac: float,
             merge_gap_min: float,
             min_points: int) -> Dict[str, float]:
    """
    [ES] RobustC FAST:
      - Elegir subset de estaciones
      - Filtrar (t_act, sta_act) por esas estaciones
    [EN] RobustC FAST:
      - Choose subset of stations
      - Filter (t_act, sta_act) by those stations
    """
    rng = np.random.default_rng(int(seed))
    k = max(1, int(round(unique_stas.size * float(keep_frac))))
    chosen = rng.choice(unique_stas, size=k, replace=False)

    mask = np.isin(sta_act, chosen)
    t_sub = t_act[mask]
    sta_sub = sta_act[mask]

    # Nota: el denominador para fracciones es k (estaciones retenidas)
    return compute_metrics_from_active(t_sub, sta_sub, k, merge_gap_min, min_points)


# ============================================================
# [ES] Plotting / [EN] Plotting
# ============================================================

def plot_hist(sim_vals: np.ndarray, real_val: float, title: str, out_png: Path) -> None:
    sim_vals = np.asarray(sim_vals, dtype=float)
    sim_vals = sim_vals[np.isfinite(sim_vals)]
    ensure_dir(out_png.parent)

    plt.figure(figsize=(10, 6))
    if sim_vals.size:
        plt.hist(sim_vals, bins=30)
    if np.isfinite(real_val):
        plt.axvline(real_val, linestyle="--")
    plt.title(title)
    plt.xlabel("metric / métrica")
    plt.ylabel("count / conteo")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ============================================================
# [ES] Parallel runner / [EN] Parallel runner
# ============================================================

def run_parallel(seeds: np.ndarray, func, n_jobs: int, desc: str, backend: str):
    """
    [ES] backend:
      - thread: ThreadPool (numpy suele liberar GIL → muy rápido en Windows)
      - process: ProcessPool (más overhead en Windows)
    [EN] backend:
      - thread: ThreadPool
      - process: ProcessPool
    """
    if n_jobs <= 1:
        out = []
        for s in tqdm(seeds, total=len(seeds), desc=desc):
            out.append(func(int(s)))
        return out

    if backend == "thread":
        out = []
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(func, int(s)) for s in seeds]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                out.append(fut.result())
        return out

    ctx = mp.get_context("spawn")
    out = []
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        futures = [ex.submit(func, int(s)) for s in seeds]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            out.append(fut.result())
    return out


# ============================================================
# [ES] Main / [EN] Main
# ============================================================

DEFAULT_METRICS = [
    "n_phases",
    "phase1_duration_h",
    "gap_1_2_h",
    "phase1_fraction_active",
    "phase2_fraction_active",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+")
    ap.add_argument("--timecol", default="time_center_iso")
    ap.add_argument("--stacol", default="station")
    ap.add_argument("--zcol", default="z")

    ap.add_argument("--zthr", type=float, default=4.0)
    ap.add_argument("--merge_gap_min", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=10)

    ap.add_argument("--nullA_n", type=int, default=10000)
    ap.add_argument("--robC_n", type=int, default=10000)
    ap.add_argument("--robC_keep_frac", type=float, default=0.5)

    ap.add_argument("--n_jobs", type=int, default=6)
    ap.add_argument("--backend", choices=["thread", "process"], default="thread",
                    help="[ES] thread suele ir más rápido en Windows / [EN] thread is often faster on Windows")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    root = project_root()
    analysis_dir = root / "resultados" / "null_tests_13_14" / "analysis"
    fig_dir = analysis_dir / "figures"
    ensure_dir(analysis_dir)
    ensure_dir(fig_dir)

    print("== Null A + Robustez C (FAST) ==")
    print(f"Root: {root}")
    print(f"Out:  {analysis_dir}")
    print(f"Events: {args.events}")
    print(f"Params: zthr={args.zthr}, merge_gap_min={args.merge_gap_min}, min_points={args.min_points}")
    print(f"NullA sims: {args.nullA_n} | RobustC sims: {args.robC_n} | n_jobs={args.n_jobs} | backend={args.backend}")
    print(f"RobC keep_frac: {args.robC_keep_frac}")
    print("Progress: tqdm=YES\n")

    rng = np.random.default_rng(int(args.seed))
    agg_rows = []

    for ev in args.events:
        print(f"[EVENT] {ev}")

        csv_path = find_event_metrics_csv(root, ev)
        df = pd.read_csv(csv_path)

        timecol, stacol, zcol = autodetect_columns(df, args.timecol, args.stacol, args.zcol)
        print(f"  [COLS] timecol={timecol}, stacol={stacol}, zcol={zcol}  (file={csv_path.name})")

        df = df.dropna(subset=[timecol, stacol, zcol]).copy()

        # ===== Precompute arrays once / Precalcular arrays una vez =====
        t_all = parse_time_to_hours(df[timecol]).astype(float)
        sta_codes, uniques = pd.factorize(df[stacol].astype(str), sort=False)  # int codes
        sta_codes = sta_codes.astype(np.int32)
        absz = np.abs(df[zcol].to_numpy(dtype=float))

        active_mask = absz >= float(args.zthr)

        t_act_full = t_all[active_mask]
        sta_act_full = sta_codes[active_mask]
        n_active = int(t_act_full.size)
        n_stations_total = int(uniques.size)

        # ===== REAL metrics =====
        real = compute_metrics_from_active(
            t_act_full,
            sta_act_full,
            n_stations_total,
            args.merge_gap_min,
            args.min_points
        )

        # write where 18 expects
        real_long = pd.DataFrame([{"metric": k, "real_value": real.get(k, np.nan)} for k in args.metrics])
        real_path = analysis_dir / f"{ev}_real_metrics.csv"
        real_long.to_csv(real_path, index=False)
        print(f"  -> wrote: {real_path}")
        print(f"  REAL phases={int(real.get('n_phases', 0))}  metrics={real}")

        agg_row = {"event": ev}
        for k in args.metrics:
            agg_row[k] = real.get(k, np.nan)
        agg_rows.append(agg_row)

        # ===== NullA =====
        seedsA = rng.integers(0, 2**31 - 1, size=int(args.nullA_n), dtype=np.int64)

        def _fA(seed: int):
            return sim_nullA(
                seed=seed,
                t_all=t_all,
                sta_act=sta_act_full,
                n_stations_total=n_stations_total,
                n_active=n_active,
                merge_gap_min=args.merge_gap_min,
                min_points=args.min_points,
            )

        print("[RUN] Null A (time-shuffle) ...")
        rowsA = run_parallel(seedsA, _fA, args.n_jobs, desc=f"NullA {ev}", backend=args.backend)
        dfA = pd.DataFrame(rowsA)
        outA = analysis_dir / f"{ev}_nullA_simulations.csv"
        dfA.to_csv(outA, index=False)
        print(f"  -> wrote: {outA}")

        # ===== RobustC =====
        seedsC = rng.integers(0, 2**31 - 1, size=int(args.robC_n), dtype=np.int64)

        # Nota: muestreamos de las estaciones que aparecen en ACTIVE; es consistente con este test.
        unique_stas = np.unique(sta_act_full)

        def _fC(seed: int):
            return sim_robC(
                seed=seed,
                t_act=t_act_full,
                sta_act=sta_act_full,
                unique_stas=unique_stas,
                keep_frac=args.robC_keep_frac,
                merge_gap_min=args.merge_gap_min,
                min_points=args.min_points,
            )

        print("[RUN] Robustez C (submuestreo estaciones) ...")
        rowsC = run_parallel(seedsC, _fC, args.n_jobs, desc=f"RobC {ev}", backend=args.backend)
        dfC = pd.DataFrame(rowsC)
        outC = analysis_dir / f"{ev}_robC_simulations.csv"
        dfC.to_csv(outC, index=False)
        print(f"  -> wrote: {outC}")

        # ===== Figures =====
        if not args.no_figures:
            for k in args.metrics:
                if k in dfA.columns:
                    plot_hist(dfA[k].to_numpy(float), real.get(k, np.nan),
                              f"{ev} — {k} (NullA)", fig_dir / f"{ev}_{k}_nullA.png")
                if k in dfC.columns:
                    plot_hist(dfC[k].to_numpy(float), real.get(k, np.nan),
                              f"{ev} — {k} (RobC)", fig_dir / f"{ev}_{k}_robC.png")

        print("")

    # aggregated real summary
    agg_df = pd.DataFrame(agg_rows)
    agg_path = analysis_dir / "real_multi_phase_summary.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[OK] REAL aggregate: {agg_path}")
    print("[OK] Finished Null A + Robustez C (FAST)")


if __name__ == "__main__":
    mp.freeze_support()
    main()
