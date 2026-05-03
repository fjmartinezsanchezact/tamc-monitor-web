
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[ES]
15_null_tests_13_14.py

Script único para:
- Análisis REAL equivalente a Test 13 (fases) + Test 14 (cobertura)
- Null test A: time-shuffle (baraja z vs tiempo)
- Null test B: t0 aleatorio (desalineación temporal)
- Robustez C: barrido de umbral zthr (p.ej. 3,4,5,6)

Incluye barras de progreso (tqdm si está disponible; si no, barra ASCII).

Lee por evento:
  resultados/<evento>/metrics/tamc_24h_metrics_allstations.csv
y, si existen, CSVs por estación:
  resultados/<evento>/metrics/tamc_24h_metrics_<STA>.csv

[EN]
Single script for:
- REAL analysis equivalent to Test 13 (phases) + Test 14 (coverage)
- Null test A: time-shuffle (shuffle z vs time)
- Null test B: random t0 (temporal misalignment)
- Robustness C: sweep zthr (e.g. 3,4,5,6)

Includes progress bars (tqdm if available; otherwise ASCII bar).

Reads per event:
  resultados/<event>/metrics/tamc_24h_metrics_allstations.csv
and if available, per-station CSVs:
  resultados/<event>/metrics/tamc_24h_metrics_<STA>.csv
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import glob
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# PROGRESS BAR (tqdm optional) / BARRA DE PROGRESO (tqdm opcional)
# ============================================================

"""
[ES]
Usamos tqdm si está instalado. Si no, usamos una barra ASCII simple.

[EN]
We use tqdm if installed. If not, we use a simple ASCII progress bar.
"""
try:
    from tqdm import tqdm as _tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    TQDM_AVAILABLE = False


def progress_iter(it, total=None, desc=""):
    """
    [ES] Iterador con barra de progreso (tqdm o ASCII).
    [EN] Iterator with progress bar (tqdm or ASCII).
    """
    if TQDM_AVAILABLE:
        return _tqdm(it, total=total, desc=desc, ncols=100, leave=False)
    return _AsciiProgress(it, total=total, desc=desc)


class _AsciiProgress:
    """
    [ES] Barra de progreso ASCII compatible con Windows CMD.
    [EN] ASCII progress bar compatible with Windows CMD.
    """
    def __init__(self, it, total=None, desc=""):
        self.it = it
        self.total = total if total is not None else None
        self.desc = desc
        self.i = 0
        if self.total is None:
            try:
                self.total = len(it)  # may fail
            except Exception:
                self.total = None

    def __iter__(self):
        for x in self.it:
            self.i += 1
            self._print()
            yield x
        self._finish()

    def _print(self):
        if self.total:
            width = 30
            frac = min(1.0, self.i / self.total)
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            msg = f"\r{self.desc} [{bar}] {self.i}/{self.total}"
        else:
            msg = f"\r{self.desc} {self.i}"
        print(msg, end="", flush=True)

    def _finish(self):
        print("")


# ============================================================
# EVENT ORIGIN TIMES (UTC) / TIEMPOS ORIGEN (UTC)
# ============================================================

EVENT_ORIGIN_UTC: Dict[str, str] = {
    "maule2010":   "2010-02-27T06:34:11Z",
    "tohoku2011":  "2011-03-11T05:46:23Z",
    "sumatra2004": "2004-12-26T00:58:53Z",
    "mexico2017":  "2017-09-19T18:14:39Z",
}


# ============================================================
# DATA STRUCTURES / ESTRUCTURAS
# ============================================================

@dataclass
class Phase:
    """
    [ES] Fase temporal: segmento continuo donde |z|>=umbral.
    [EN] Temporal phase: continuous segment where |z|>=threshold.
    """
    start_h: float
    end_h: float
    n_points: int
    peak_absz: float
    mean_absz: float

    @property
    def duration_h(self) -> float:
        return float(self.end_h - self.start_h)


# ============================================================
# PATH HELPERS / RUTAS
# ============================================================

def guess_project_root(cwd: str) -> str:
    """
    [ES] Asume ejecución desde <repo>/src/core y sube dos niveles.
    [EN] Assumes execution from <repo>/src/core and goes up two levels.
    """
    return os.path.abspath(os.path.join(cwd, "..", ".."))


def metrics_dir(root: str, event: str) -> str:
    return os.path.join(root, "resultados", event, "metrics")


def allstations_csv_path(root: str, event: str) -> str:
    return os.path.join(metrics_dir(root, event), "tamc_24h_metrics_allstations.csv")


def station_csv_paths(root: str, event: str) -> List[str]:
    """
    [ES] CSVs por estación (excluye allstations). Si no hay, lista vacía.
    [EN] Per-station CSVs (excluding allstations). If none, empty list.
    """
    mdir = metrics_dir(root, event)
    paths = sorted(glob.glob(os.path.join(mdir, "tamc_24h_metrics_*.csv")))
    paths = [p for p in paths if os.path.basename(p).lower() != "tamc_24h_metrics_allstations.csv"]
    return paths


# ============================================================
# COLUMN DETECTION / DETECCIÓN DE COLUMNAS
# ============================================================

def pick_z_col(df: pd.DataFrame) -> str:
    """
    [ES] Detecta columna de z-score.
    [EN] Detect z-score column.
    """
    for c in ["zscore", "z_score", "z", "Z", "zScore"]:
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna zscore/z_score. Columnas: {list(df.columns)}")


def _infer_t0_from_event_name(event: str):
    """Infer t0 from event folder name pattern ..._YYYYMMDD_HHMMSS (UTC)."""
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
    [ES] Garantiza columna 't_hours' sin depender de EVENT_ORIGIN_UTC.
    Prioridad:
      1) Columnas ya en horas (t_hours/time_center_h/time_h/...)
      2) time_center_iso + t0 inferido del nombre del evento (YYYYMMDD_HHMMSS)
      3) time_center_iso + t0 inferido del propio CSV (max |zscore| si existe, si no primer timestamp válido)

    [EN] Ensure 't_hours' without requiring EVENT_ORIGIN_UTC.
    """
    # Already present
    if "t_hours" in df.columns:
        return df

    # 1) hour-like columns
    hour_like = [
        "t_hours", "t_hour", "t_h",
        "time_center_h", "time_h",
        "hours_from_event", "t_rel_hours", "t_rel_h",
        "hours_rel", "hours", "hour",
    ]
    for c in hour_like:
        if c in df.columns:
            out = df.copy()
            out["t_hours"] = pd.to_numeric(out[c], errors="coerce")
            return out

    # Need ISO timestamps
    if "time_center_iso" not in df.columns:
        raise ValueError("Falta time_center_iso y no hay columna en horas (t_hours).")

    out = df.copy()
    s = out["time_center_iso"].astype(str).str.strip()
    s = s.str.replace("Z", "+00:00", regex=False)

    try:
        t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except TypeError:
        t = pd.to_datetime(s, errors="coerce", utc=True)

    # Try to fill remaining NaT with non-UTC parsing
    if t.isna().any():
        try:
            t2 = pd.to_datetime(s, errors="coerce", format="mixed")
        except TypeError:
            t2 = pd.to_datetime(s, errors="coerce")

        if getattr(t2.dt, "tz", None) is not None:
            t2 = t2.dt.tz_convert("UTC")
        else:
            t2 = t2.dt.tz_localize("UTC")

        t = t.fillna(t2)

    if t.isna().any():
        bad = out.loc[t.isna(), "time_center_iso"].head(10).tolist()
        raise ValueError(f"time_center_iso no parseable (ejemplos): {bad}")

    # 2) t0 from event name
    t0 = _infer_t0_from_event_name(event)

    # 3) fallback from data (max |zscore|, else first timestamp)
    if t0 is None:
        if "zscore" in out.columns:
            z = pd.to_numeric(out["zscore"], errors="coerce")
            if z.notna().any():
                idx = (z.abs()).idxmax()
                try:
                    t0 = t.loc[idx]
                except Exception:
                    t0 = None
        if t0 is None:
            t_valid = t.dropna()
            if len(t_valid) == 0:
                raise ValueError(f"No puedo parsear time_center_iso para '{event}'.")
            t0 = t_valid.iloc[0]

    out["t_hours"] = (t - t0).dt.total_seconds() / 3600.0
    return out


# ============================================================
# PHASE DETECTION / DETECCIÓN DE FASES
# ============================================================

def detect_phases_from_series(t_hours: np.ndarray,
                              absz: np.ndarray,
                              zthr: float,
                              merge_gap_min: float,
                              min_points: int) -> List[Phase]:
    """
    [ES] Detecta fases: |z|>=zthr, une si gap<=merge_gap_min, exige min_points.
    [EN] Detect phases: |z|>=zthr, merge if gap<=merge_gap_min, require min_points.
    """
    mask = np.isfinite(t_hours) & np.isfinite(absz)
    t = t_hours[mask]
    a = absz[mask]
    if t.size == 0:
        return []

    order = np.argsort(t)
    t = t[order]
    a = a[order]

    idx = np.where(a >= zthr)[0]
    if idx.size == 0:
        return []

    gap_h = merge_gap_min / 60.0
    phases: List[Phase] = []

    seg = [int(idx[0])]
    for k in idx[1:]:
        k = int(k)
        if (t[k] - t[seg[-1]]) <= gap_h:
            seg.append(k)
        else:
            if len(seg) >= min_points:
                phases.append(_build_phase(t, a, seg))
            seg = [k]

    if len(seg) >= min_points:
        phases.append(_build_phase(t, a, seg))

    return phases


def _build_phase(t: np.ndarray, a: np.ndarray, seg: List[int]) -> Phase:
    th = t[seg]
    az = a[seg]
    return Phase(
        start_h=float(np.min(th)),
        end_h=float(np.max(th)),
        n_points=int(len(seg)),
        peak_absz=float(np.max(az)),
        mean_absz=float(np.mean(az)),
    )


# ============================================================
# CORE ANALYSIS / ANÁLISIS PRINCIPAL
# ============================================================

def load_allstations(root: str, event: str) -> Tuple[pd.DataFrame, str]:
    p = allstations_csv_path(root, event)
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_csv(p)
    zcol = pick_z_col(df)
    df = ensure_t_hours(df, event)
    return df, zcol


def compute_phases_allstations(df_all: pd.DataFrame, zcol: str,
                              zthr: float, merge_gap_min: float, min_points: int) -> List[Phase]:
    absz = np.abs(pd.to_numeric(df_all[zcol], errors="coerce").to_numpy())
    t_hours = pd.to_numeric(df_all["t_hours"], errors="coerce").to_numpy()
    return detect_phases_from_series(t_hours, absz, zthr, merge_gap_min, min_points)


def compute_station_coverage(root: str, event: str, phases: List[Phase],
                             zthr: float, max_phases: int = 2) -> List[Dict]:
    """
    [ES]
    Cobertura: usa CSVs por estación si existen.
    Una estación se considera activa si existe al menos 1 punto dentro de la fase con |z|>=zthr.

    [EN]
    Coverage: uses per-station CSVs if available.
    A station is active if it has at least 1 point inside the phase with |z|>=zthr.
    """
    paths = station_csv_paths(root, event)

    # Sin CSVs por estación => devolvemos NaN (informativo)
    if len(paths) == 0:
        out = []
        for i, ph in enumerate(phases[:max_phases], start=1):
            out.append({
                "event": event,
                "phase": i,
                "phase_start_h": ph.start_h,
                "phase_end_h": ph.end_h,
                "phase_duration_h": ph.duration_h,
                "stations_total": np.nan,
                "stations_active": np.nan,
                "fraction_active": np.nan,
            })
        return out

    # Pre-carga
    station_data = []
    for sp in paths:
        df = pd.read_csv(sp)
        zcol = pick_z_col(df)
        df = ensure_t_hours(df, event)
        t = pd.to_numeric(df["t_hours"], errors="coerce").to_numpy()
        a = np.abs(pd.to_numeric(df[zcol], errors="coerce").to_numpy())
        station_data.append((t, a))

    stations_total = len(station_data)
    results: List[Dict] = []

    for i, ph in enumerate(phases[:max_phases], start=1):
        active = 0
        for t, a in station_data:
            mask = np.isfinite(t) & np.isfinite(a)
            tt = t[mask]
            aa = a[mask]
            if tt.size == 0:
                continue
            inwin = (tt >= ph.start_h) & (tt <= ph.end_h)
            if np.any(aa[inwin] >= zthr):
                active += 1

        results.append({
            "event": event,
            "phase": i,
            "phase_start_h": ph.start_h,
            "phase_end_h": ph.end_h,
            "phase_duration_h": ph.duration_h,
            "stations_total": stations_total,
            "stations_active": active,
            "fraction_active": (active / stations_total) if stations_total > 0 else np.nan,
        })

    return results


# ============================================================
# NULL MODELS / MODELOS NULOS
# ============================================================

def nullA_time_shuffle(df: pd.DataFrame, zcol: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    [ES] Null A: baraja z (rompe relación temporal) manteniendo distribución.
    [EN] Null A: shuffle z (break time structure) preserving distribution.
    """
    out = df.copy()
    z = pd.to_numeric(out[zcol], errors="coerce").to_numpy()
    idx = np.arange(len(z))
    rng.shuffle(idx)
    out[zcol] = z[idx]
    return out


def nullB_random_t0_shift(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, float]:
    """
    [ES] Null B: shift aleatorio en t_hours (equivale a t0 aleatorio).
    [EN] Null B: random shift in t_hours (equivalent to random t0).
    """
    out = df.copy()
    t = pd.to_numeric(out["t_hours"], errors="coerce").to_numpy()
    t = t[np.isfinite(t)]
    if t.size < 10:
        return out, 0.0
    tmin, tmax = float(np.min(t)), float(np.max(t))
    shift = rng.uniform(tmin, tmax)
    out["t_hours"] = pd.to_numeric(out["t_hours"], errors="coerce") - shift
    return out, float(shift)


# ============================================================
# RUNNERS / EJECUCIÓN
# ============================================================

def run_real(root: str, events: List[str], zthr: float, merge_gap_min: float, min_points: int,
             outdir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    [ES] Corre análisis real y guarda CSVs.
    [EN] Run real analysis and save CSVs.
    """
    rows_sum = []
    rows_cov = []

    for ev in progress_iter(events, total=len(events), desc="[REAL] eventos"):
        df_all, zcol = load_allstations(root, ev)
        phases = compute_phases_allstations(df_all, zcol, zthr, merge_gap_min, min_points)

        row = {
            "event": ev,
            "metrics_csv": allstations_csv_path(root, ev),
            "zcol": zcol,
            "zthr": zthr,
            "merge_gap_min": merge_gap_min,
            "min_points": min_points,
            "n_phases": float(len(phases)),
        }
        if len(phases) >= 1:
            p1 = phases[0]
            row.update({
                "phase1_start_h": p1.start_h,
                "phase1_end_h": p1.end_h,
                "phase1_duration_h": p1.duration_h,
                "phase1_n": float(p1.n_points),
                "phase1_peak_absz": p1.peak_absz,
                "phase1_mean_absz": p1.mean_absz,
            })
        if len(phases) >= 2:
            p2 = phases[1]
            row.update({
                "phase2_start_h": p2.start_h,
                "phase2_end_h": p2.end_h,
                "phase2_duration_h": p2.duration_h,
                "phase2_n": float(p2.n_points),
                "phase2_peak_absz": p2.peak_absz,
                "phase2_mean_absz": p2.mean_absz,
                "gap_1_2_h": p2.start_h - phases[0].end_h,
            })
        rows_sum.append(row)

        cov_rows = compute_station_coverage(root, ev, phases, zthr=zthr, max_phases=2)
        for cr in cov_rows:
            cr["metrics_csv"] = allstations_csv_path(root, ev)
            cr["zcol"] = zcol
            rows_cov.append(cr)

    df_sum = pd.DataFrame(rows_sum)
    df_cov = pd.DataFrame(rows_cov)

    os.makedirs(outdir, exist_ok=True)
    df_sum.to_csv(os.path.join(outdir, "real_multi_phase_summary.csv"), index=False)
    df_cov.to_csv(os.path.join(outdir, "real_station_coverage_by_phase.csv"), index=False)

    return df_sum, df_cov


def run_nullA(root: str, events: List[str], zthr: float, merge_gap_min: float, min_points: int,
              n_sims: int, seed: int, outdir: str) -> pd.DataFrame:
    """
    [ES] Null A: barajado z vs tiempo. Repite n_sims por evento.
    [EN] Null A: shuffle z vs time. Repeats n_sims per event.
    """
    rng = np.random.default_rng(seed)
    rows = []

    total_steps = len(events) * n_sims
    step = 0

    for ev in events:
        df_all, zcol = load_allstations(root, ev)
        # progreso por simulaciones del evento
        for s in progress_iter(range(n_sims), total=n_sims, desc=f"[Null A] {ev}"):
            dnull = nullA_time_shuffle(df_all, zcol, rng=rng)
            phases = compute_phases_allstations(dnull, zcol, zthr, merge_gap_min, min_points)

            row = {
                "null_model": "A_time_shuffle",
                "event": ev,
                "sim": s,
                "zthr": zthr,
                "merge_gap_min": merge_gap_min,
                "min_points": min_points,
                "n_phases": float(len(phases)),
                "phase1_duration_h": phases[0].duration_h if len(phases) >= 1 else np.nan,
                "phase1_peak_absz": phases[0].peak_absz if len(phases) >= 1 else np.nan,
            }

            # cobertura null coherente (si hay CSVs por estación)
            paths = station_csv_paths(root, ev)
            if len(paths) == 0 or len(phases) == 0:
                row["phase2_fraction_active"] = np.nan
            else:
                station_data = []
                for sp in paths:
                    sdf = pd.read_csv(sp)
                    sz = pick_z_col(sdf)
                    sdf = ensure_t_hours(sdf, ev)
                    sdf = nullA_time_shuffle(sdf, sz, rng=rng)
                    t = pd.to_numeric(sdf["t_hours"], errors="coerce").to_numpy()
                    a = np.abs(pd.to_numeric(sdf[sz], errors="coerce").to_numpy())
                    station_data.append((t, a))

                ph = phases[1] if len(phases) >= 2 else phases[0]
                active = 0
                total = len(station_data)
                for t, a in station_data:
                    mask = np.isfinite(t) & np.isfinite(a)
                    tt = t[mask]; aa = a[mask]
                    inwin = (tt >= ph.start_h) & (tt <= ph.end_h)
                    if np.any(aa[inwin] >= zthr):
                        active += 1
                row["phase2_fraction_active"] = (active / total) if total > 0 else np.nan

            rows.append(row)
            step += 1

    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "nullA_simulations.csv"), index=False)
    return df


def run_nullB(root: str, events: List[str], zthr: float, merge_gap_min: float, min_points: int,
              n_sims: int, seed: int, outdir: str) -> pd.DataFrame:
    """
    [ES] Null B: t0 aleatorio (shift t_hours). Repite n_sims por evento.
    [EN] Null B: random t0 (shift t_hours). Repeats n_sims per event.
    """
    rng = np.random.default_rng(seed + 12345)
    rows = []

    for ev in events:
        df_all, zcol = load_allstations(root, ev)
        for s in progress_iter(range(n_sims), total=n_sims, desc=f"[Null B] {ev}"):
            dnull, shift = nullB_random_t0_shift(df_all, rng=rng)
            phases = compute_phases_allstations(dnull, zcol, zthr, merge_gap_min, min_points)

            row = {
                "null_model": "B_random_t0",
                "event": ev,
                "sim": s,
                "shift_h": shift,
                "zthr": zthr,
                "merge_gap_min": merge_gap_min,
                "min_points": min_points,
                "n_phases": float(len(phases)),
                "phase1_start_h": phases[0].start_h if len(phases) >= 1 else np.nan,
                "phase1_duration_h": phases[0].duration_h if len(phases) >= 1 else np.nan,
            }

            # cobertura con t0 aleatorio: shift también en estaciones
            paths = station_csv_paths(root, ev)
            if len(paths) == 0 or len(phases) == 0:
                row["phase2_fraction_active"] = np.nan
            else:
                station_data = []
                for sp in paths:
                    sdf = pd.read_csv(sp)
                    sz = pick_z_col(sdf)
                    sdf = ensure_t_hours(sdf, ev).copy()
                    sdf["t_hours"] = pd.to_numeric(sdf["t_hours"], errors="coerce") - shift
                    t = pd.to_numeric(sdf["t_hours"], errors="coerce").to_numpy()
                    a = np.abs(pd.to_numeric(sdf[sz], errors="coerce").to_numpy())
                    station_data.append((t, a))

                ph = phases[1] if len(phases) >= 2 else phases[0]
                active = 0
                total = len(station_data)
                for t, a in station_data:
                    mask = np.isfinite(t) & np.isfinite(a)
                    tt = t[mask]; aa = a[mask]
                    inwin = (tt >= ph.start_h) & (tt <= ph.end_h)
                    if np.any(aa[inwin] >= zthr):
                        active += 1
                row["phase2_fraction_active"] = (active / total) if total > 0 else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "nullB_simulations.csv"), index=False)
    return df


def run_C_robustness(root: str, events: List[str], zthr_list: List[float],
                     merge_gap_min: float, min_points: int, outdir: str) -> pd.DataFrame:
    """
    [ES] Robustez C: repite análisis real para varios zthr.
    [EN] Robustness C: repeats real analysis across multiple zthr values.
    """
    rows = []
    os.makedirs(outdir, exist_ok=True)

    for zthr in progress_iter(zthr_list, total=len(zthr_list), desc="[C] zthr sweep"):
        subdir = os.path.join(outdir, f"zthr_{zthr:g}")
        df_sum, df_cov = run_real(root, events, zthr, merge_gap_min, min_points, outdir=subdir)

        for ev in events:
            sub = df_sum[df_sum["event"] == ev]
            cov = df_cov[df_cov["event"] == ev]
            # phase2 fraction_active si existe
            p2 = cov[cov["phase"] == 2]["fraction_active"]
            rows.append({
                "event": ev,
                "zthr": zthr,
                "n_phases": float(sub["n_phases"].iloc[0]) if len(sub) else np.nan,
                "phase1_duration_h": float(sub["phase1_duration_h"].iloc[0]) if ("phase1_duration_h" in sub.columns and len(sub)) else np.nan,
                "phase2_fraction_active": float(p2.iloc[0]) if len(p2) else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "C_robustness_summary.csv"), index=False)
    return df


# ============================================================
# MAIN / ENTRADA
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="[ES/EN] Null tests A/B + robustness C for (Test 13 + Test 14) in one script, with progress bars."
    )
    ap.add_argument("events", nargs="+",
                    help="Event IDs, e.g.: maule2010 tohoku2011 sumatra2004 mexico2017")

    ap.add_argument("--root", default=None, help="Project root (default: auto from src/core)")
    ap.add_argument("--outdir", default=None, help="Output dir (default: resultados/null_tests_13_14)")

    ap.add_argument("--zthr", type=float, default=4.0, help="Main threshold for REAL + null A/B")
    ap.add_argument("--merge_gap_min", type=float, default=6.0)
    ap.add_argument("--min_points", type=int, default=10)

    ap.add_argument("--n_sims", type=int, default=200, help="Simulations per event for null A and B")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")

    ap.add_argument("--zthr_list", default="3,4,5,6",
                    help="Comma-separated thresholds for robustness C (e.g. 3,4,5,6)")

    args = ap.parse_args()

    root = args.root or guess_project_root(os.getcwd())
    outdir = args.outdir or os.path.join(root, "resultados", "null_tests_13_14")
    os.makedirs(outdir, exist_ok=True)

    events = args.events
    zthr = float(args.zthr)
    merge_gap_min = float(args.merge_gap_min)
    min_points = int(args.min_points)
    n_sims = int(args.n_sims)
    seed = int(args.seed)
    zthr_list = [float(x.strip()) for x in args.zthr_list.split(",") if x.strip()]

    print("== 15_null_tests_13_14 ==")
    print(f"Root:   {root}")
    print(f"Outdir: {outdir}")
    print(f"Events: {events}")
    print(f"Params: zthr={zthr}, merge_gap_min={merge_gap_min}, min_points={min_points}")
    print(f"Null:   n_sims={n_sims}, seed={seed}")
    print(f"C:      zthr_list={zthr_list}")
    print(f"Progress: tqdm={'YES' if TQDM_AVAILABLE else 'NO (ASCII)'}\n")

    # REAL
    print("[RUN] REAL (Test 13 + Test 14) ...")
    run_real(root, events, zthr, merge_gap_min, min_points, outdir=outdir)
    print("  -> wrote: real_multi_phase_summary.csv, real_station_coverage_by_phase.csv\n")

    # Null A
    print("[RUN] Null A (time-shuffle) ...")
    run_nullA(root, events, zthr, merge_gap_min, min_points, n_sims, seed, outdir=outdir)
    print("  -> wrote: nullA_simulations.csv\n")

    # Null B
    print("[RUN] Null B (random t0) ...")
    run_nullB(root, events, zthr, merge_gap_min, min_points, n_sims, seed, outdir=outdir)
    print("  -> wrote: nullB_simulations.csv\n")

    # Robustness C
    print("[RUN] Robustness C (zthr sweep) ...")
    run_C_robustness(root, events, zthr_list, merge_gap_min, min_points, outdir=os.path.join(outdir, "C_robustness"))
    print("  -> wrote: C_robustness_summary.csv (and per-zthr real outputs)\n")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
