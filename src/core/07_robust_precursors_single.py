"""
07_robust_precursors_single.py

Igual que tu versión, pero:
- mejora pequeña en auto-log del panel mean_absz (más robusto, etiqueta)
"""

from __future__ import annotations

import argparse
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from datetime import datetime
from obspy import UTCDateTime
import pandas as pd

import matplotlib


def infer_mainshock_time_from_name(name: str) -> UTCDateTime | None:
    """
    Infer a timestamp from strings like:
      ..._YYYYMMDD_HHMMSS
    Works for both event and control folder names.
    """
    base = name.replace("\\", "/").strip("/")
    # If path like EVENT/control_XX_YYYYMMDD_HHMMSS, use the last component for control time
    tail = base.split("/")[-1]
    m = re.search(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)", tail)
    if not m:
        # try full string (event name may have the timestamp earlier)
        m = re.search(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)", base)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return UTCDateTime(dt)
        except Exception:
            return None
    return None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

def _to_pd_utc(ts):
    """Convert ObsPy UTCDateTime / datetime / pandas-like to pandas Timestamp (UTC)."""
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    if isinstance(ts, UTCDateTime):
        return pd.Timestamp(ts.datetime, tz="UTC")
    return pd.to_datetime(ts, utc=True)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config helpers
# ----------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_events_csv(events_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(events_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_event_column(df: pd.DataFrame) -> str:
    for c in ["event", "evento", "name", "event_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"events.csv no tiene columna de evento. Columnas: {list(df.columns)}")


def find_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "origin_time", "origin_time_utc",
        "mainshock_time", "mainshock_time_utc",
        "time", "time_utc", "datetime", "date_time",
        "fecha", "hora", "fecha_hora"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in ["origin", "mainshock", "time", "date", "fecha", "hora"]):
            return c
    raise KeyError(f"events.csv no tiene columna de tiempo/origen. Columnas: {list(df.columns)}")


def load_mainshock_time(events_csv: Path, event: str) -> pd.Timestamp:
    df = read_events_csv(events_csv)
    evcol = find_event_column(df)
    tcol = find_time_column(df)

    row = df[df[evcol].astype(str) == str(event)]
    if row.empty:
        ms = infer_mainshock_time_from_name(event)
        if ms is not None:
            print(f"[WARN] Evento '{event}' no está en {events_csv}. Uso fallback desde nombre: {ms}.")
            return ms
        raise ValueError(f"Evento '{event}' no encontrado en {events_csv} y no pude inferir fecha/hora del nombre")

    t = pd.to_datetime(row.iloc[0][tcol], utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"No pude parsear mainshock time de events.csv columna '{tcol}' para evento '{event}'")
    return t


def ensure_utc(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, utc=True, errors="coerce")


# ----------------------------
# Core computations
# ----------------------------

@dataclass
class Episode:
    start_h: float
    end_h: float
    duration_min: float
    max_f: float


def build_timeseries(df: pd.DataFrame,
                     mainshock: pd.Timestamp,
                     hours: float,
                     bin_min: int,
                     z0: float) -> Tuple[pd.DataFrame, List[str]]:
    if "time_center_iso" not in df.columns:
        raise KeyError("metrics_allstations debe tener columna 'time_center_iso'")
    if "station_id" not in df.columns:
        raise KeyError("metrics_allstations debe tener columna 'station_id'")

    zcol = None
    for cand in ["zscore", "z_score", "z", "zmax", "z_max"]:
        if cand in df.columns:
            zcol = cand
            break
    if zcol is None:
        raise KeyError(f"No encuentro columna zscore en metrics_allstations. Columnas: {list(df.columns)}")

    df = df.copy()
    df["time"] = ensure_utc(df["time_center_iso"])
    df = df.dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["t_rel_h"] = (df["time"] - mainshock).dt.total_seconds() / 3600.0

    df = df[(df["t_rel_h"] >= -hours) & (df["t_rel_h"] <= 0)].copy()
    if df.empty:
        raise ValueError(f"No hay datos en ventana [-{hours}, 0]h")

    df["absz"] = df[zcol].astype(float).abs()

    bin_td = pd.Timedelta(minutes=bin_min)
    df["bin_start"] = df["time"].dt.floor(f"{bin_min}min")
    df["bin_center"] = df["bin_start"] + bin_td/2

    stations = sorted(df["station_id"].astype(str).unique().tolist())

    g = df.groupby(["bin_center", "station_id"])["absz"].mean().reset_index()

    def agg_bin(x: pd.DataFrame) -> pd.Series:
        absz_vals = x["absz"].to_numpy()
        active = (absz_vals >= z0).sum()
        n = len(absz_vals)
        return pd.Series({
            "f_active": float(active) / float(n) if n else np.nan,
            "mean_absz": float(np.nanmean(absz_vals)) if n else np.nan,
            "n_stations": int(n),
        })

    ts = g.groupby("bin_center", as_index=False).apply(agg_bin, include_groups=False)
    ts["bin_center"] = pd.to_datetime(ts["bin_center"], utc=True, errors="coerce")
    ts["t_rel_h"] = (ts["bin_center"] - mainshock).dt.total_seconds() / 3600.0
    ts = ts.sort_values("t_rel_h")

    return ts, stations


def detect_episodes(ts: pd.DataFrame, f0: float, min_dur_min: int) -> List[Episode]:
    if ts.empty:
        return []

    bin_minutes = np.median(np.diff(ts["t_rel_h"].to_numpy())) * 60
    if not np.isfinite(bin_minutes) or bin_minutes <= 0:
        if len(ts) >= 2:
            bin_minutes = (ts["bin_center"].iloc[1] - ts["bin_center"].iloc[0]).total_seconds() / 60.0
        else:
            bin_minutes = float(min_dur_min)

    mask = ts["f_active"].to_numpy() >= f0
    episodes: List[Episode] = []
    i = 0
    while i < len(ts):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(ts) and mask[j]:
            j += 1
        dur = (j - i) * bin_minutes
        if dur >= min_dur_min:
            start_h = float(ts["t_rel_h"].iloc[i])
            end_h = float(ts["t_rel_h"].iloc[j-1])
            max_f = float(ts["f_active"].iloc[i:j].max())
            episodes.append(Episode(start_h=start_h, end_h=end_h, duration_min=float(dur), max_f=max_f))
        i = j
    return episodes


def circular_shift_null(df: pd.DataFrame,
                        mainshock: pd.Timestamp,
                        hours: float,
                        bin_min: int,
                        z0: float,
                        f0: float,
                        min_dur_min: int,
                        mc: int,
                        seed: int = 123) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    df0 = df.copy()
    df0["time"] = ensure_utc(df0["time_center_iso"])
    df0 = df0.dropna(subset=["time"])
    df0["time"] = pd.to_datetime(df0["time"], utc=True, errors="coerce")
    df0["t_rel_h"] = (df0["time"] - mainshock).dt.total_seconds() / 3600.0
    df0 = df0[(df0["t_rel_h"] >= -hours) & (df0["t_rel_h"] <= 0)].copy()
    if df0.empty:
        raise ValueError("No hay datos para construir nulo.")

    zcol = None
    for cand in ["zscore", "z_score", "z", "zmax", "z_max"]:
        if cand in df0.columns:
            zcol = cand
            break
    if zcol is None:
        raise KeyError("No encuentro columna zscore en metrics_allstations.")

    df0["absz"] = df0[zcol].astype(float).abs()

    bin_td = pd.Timedelta(minutes=bin_min)
    df0["bin_start"] = df0["time"].dt.floor(f"{bin_min}min")
    df0["bin_center"] = df0["bin_start"] + bin_td/2

    g = df0.groupby(["bin_center", "station_id"])["absz"].mean().reset_index()
    pivot = g.pivot(index="bin_center", columns="station_id", values="absz").sort_index()
    bins = pivot.index

    null_rows = []
    mat0 = pivot.to_numpy(copy=True)

    for k in range(mc):
        mat = mat0.copy()
        for si in range(mat.shape[1]):
            col = mat[:, si]
            if np.all(np.isnan(col)):
                continue
            shift = int(rng.integers(0, len(col)))
            mat[:, si] = np.roll(col, shift)

        active = (mat >= z0).astype(float)
        n = np.sum(~np.isnan(mat), axis=1)
        f_active = np.where(n > 0, np.nansum(active, axis=1) / n, np.nan)

        ts_null = pd.DataFrame({"bin_center": bins, "f_active": f_active})
        ts_null["bin_center"] = pd.to_datetime(ts_null["bin_center"], utc=True, errors="coerce")
        ts_null["t_rel_h"] = (ts_null["bin_center"] - mainshock).dt.total_seconds() / 3600.0
        ts_null = ts_null.sort_values("t_rel_h")

        episodes = detect_episodes(ts_null.assign(bin_center=ts_null["bin_center"]), f0=f0, min_dur_min=min_dur_min)
        null_rows.append({
            "mc": k,
            "max_f_active": float(np.nanmax(f_active)),
            "n_episodes": int(len(episodes)),
            "max_episode_duration_min": float(max([e.duration_min for e in episodes], default=0.0)),
        })

    null_df = pd.DataFrame(null_rows)

    summary = {
        "p95_max_f": float(null_df["max_f_active"].quantile(0.95)),
        "p99_max_f": float(null_df["max_f_active"].quantile(0.99)),
        "p95_n_episodes": float(null_df["n_episodes"].quantile(0.95)),
        "p99_n_episodes": float(null_df["n_episodes"].quantile(0.99)),
        "p95_max_dur": float(null_df["max_episode_duration_min"].quantile(0.95)),
        "p99_max_dur": float(null_df["max_episode_duration_min"].quantile(0.99)),
    }
    return null_df, summary


def p_value(null_series: pd.Series, observed: float, tail: str = "ge") -> float:
    x = null_series.to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if tail == "ge":
        return float((np.sum(x >= observed) + 1) / (len(x) + 1))
    if tail == "le":
        return float((np.sum(x <= observed) + 1) / (len(x) + 1))
    raise ValueError("tail debe ser 'ge' o 'le'")


def maybe_use_log(y: np.ndarray, ratio: float = 30.0, min_max: float = 10.0) -> bool:
    """Auto-log para series >0: usa log si max/p95 es muy grande."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    if len(y) == 0:
        return False
    p95 = np.percentile(y, 95)
    mx = np.max(y)
    if p95 <= 0:
        return False
    return (mx > min_max) and ((mx / p95) > ratio)


def make_figure(event: str,
                ts: pd.DataFrame,
                episodes: List[Episode],
                null_bands: Optional[Dict[str, float]],
                out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1.0]})

    x = ts["t_rel_h"].to_numpy()
    f = ts["f_active"].to_numpy()
    m = ts["mean_absz"].to_numpy()

    # Panel 1: f_active
    ax1.plot(x, f, lw=2.2)
    ax1.axvline(0, ls="--", lw=1.6, color="black")
    ax1.set_ylabel("Fracción de estaciones activas f(t)")
    ax1.set_title(f"Precursores multies­tación (robusto) — {event}")

    if null_bands is not None:
        ax1.axhline(null_bands["p95_max_f"], ls="--", lw=1.2)
        ax1.axhline(null_bands["p99_max_f"], ls=":", lw=1.6)
        ax1.text(0.01, 0.95,
                 f"Null p95(max f)={null_bands['p95_max_f']:.2f} | p99={null_bands['p99_max_f']:.2f}",
                 transform=ax1.transAxes, va="top")

    for e in episodes:
        ax1.axvspan(e.start_h, e.end_h, alpha=0.15)
        ax1.text((e.start_h + e.end_h)/2, ax1.get_ylim()[1]*0.92,
                 f"{e.duration_min:.0f} min\nmax f={e.max_f:.2f}",
                 ha="center", va="top", fontsize=8)

    ax1.set_ylim(0, min(1.0, np.nanmax(f)*1.15 if np.isfinite(np.nanmax(f)) else 1.0))
    ax1.grid(True, alpha=0.3)

    # Panel 2: mean_absz
    ax2.plot(x, m, lw=2.0)
    ax2.axvline(0, ls="--", lw=1.6, color="black")
    ax2.set_ylabel("Media de |z| entre estaciones")
    ax2.set_xlabel("Horas respecto al mainshock (t = 0)")

    if maybe_use_log(m, ratio=30.0, min_max=10.0):
        eps = max(1e-3, np.percentile(m[np.isfinite(m) & (m > 0)], 1) * 0.5) if np.any((m > 0) & np.isfinite(m)) else 1e-3
        ax2.set_yscale("log")
        ax2.set_ylim(eps, np.nanmax(m)*1.2)
        ax2.set_ylabel(ax2.get_ylabel() + " (log)")

    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(float(np.nanmin(x)), 0.0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("event", type=str, help="Nombre de evento (carpeta en data/resultados)")
    p.add_argument("--hours", type=float, default=24.0, help="Ventana en horas antes del mainshock (default 24)")
    p.add_argument("--bin-min", type=int, default=10, help="Bin temporal en minutos (default 10)")
    p.add_argument("--z0", type=float, default=3.0, help="Umbral |z| para actividad por estación (default 3)")
    p.add_argument("--f0", type=float, default=0.30, help="Umbral fracción activa para episodio (default 0.30)")
    p.add_argument("--min-dur", type=int, default=30, help="Duración mínima episodio (min) (default 30)")
    p.add_argument("--mc", type=int, default=500, help="Monte Carlo (circular shifts) (default 500)")
    p.add_argument("--seed", type=int, default=123, help="Semilla RNG (default 123)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    event = args.event

    root = project_root()
    metrics_all = root / "resultados" / event / "metrics" / "tamc_24h_metrics_allstations.csv"
    events_csv = root / "src" / "events.csv"

    if not metrics_all.exists():
        raise FileNotFoundError(f"No existe: {metrics_all}")

    # events.csv es OPCIONAL (para robustez en instalaciones sin catálogo local).
    # Si no existe, inferimos t0 desde el nombre del evento (…_YYYYMMDD_HHMMSS).
    have_events_csv = events_csv.exists()
    if not have_events_csv:
        print(f"[WARN] No existe: {events_csv} -> usando fallback por nombre de carpeta.")
    # Normaliza path (permite EVENTO, EVENTO/mainshock o EVENTO/control_XX_YYYYMMDD_HHMMSS)
    event_norm = event.replace("\\", "/").strip("/")
    parts = [p for p in event_norm.split("/") if p]
    if not parts:
        raise ValueError("Evento vacío")
    event_key = parts[0]
    tail = parts[-1]


    # t=0 (mainshock) depende del tipo de carpeta:
    # - Si el path apunta a un control (…/control_XX_YYYYMMDD_HHMMSS), usamos ese timestamp como t=0.
    # - Si apunta al evento base (event_key), usamos events.csv si existe, o inferimos desde el nombre.
    #
    # Nota: usar t=0 del evento base cuando se analizan controles vacía la ventana temporal y rompe la comparación.
    if tail.startswith("control_"):
        ms = infer_mainshock_time_from_name(tail)
        if ms is None:
            raise ValueError(
                f"No puedo inferir mainshock para control tail={tail!r}. "
                f"El control debe incluir _YYYYMMDD_HHMMSS."
            )
        mainshock = _to_pd_utc(ms)
    else:
        if have_events_csv:
            try:
                mainshock = load_mainshock_time(events_csv, event_key)
                mainshock = _to_pd_utc(mainshock)
            except Exception as e:
                print(f"[WARN] No pude leer mainshock desde events.csv ({e}). Intento fallback por nombre...")
                ms = infer_mainshock_time_from_name(event_key)
                if ms is None:
                    raise
                mainshock = _to_pd_utc(ms)
        else:
            ms = infer_mainshock_time_from_name(event_key)
            if ms is None:
                raise ValueError(
                    f"No puedo inferir mainshock para event_key={event_key!r}. "
                    f"Necesito src/events.csv o un nombre con _YYYYMMDD_HHMMSS."
                )
            mainshock = _to_pd_utc(ms)

    df = pd.read_csv(metrics_all)

    ts, _stations = build_timeseries(df, mainshock, hours=args.hours, bin_min=args.bin_min, z0=args.z0)
    episodes = detect_episodes(ts.assign(bin_center=ts["bin_center"]), f0=args.f0, min_dur_min=args.min_dur)

    null_df, null_summary = circular_shift_null(
        df=df, mainshock=mainshock, hours=args.hours, bin_min=args.bin_min,
        z0=args.z0, f0=args.f0, min_dur_min=args.min_dur, mc=args.mc, seed=args.seed
    )

    obs = {
        "event": event,
        "mainshock_utc": str(mainshock),
        "hours": args.hours,
        "bin_min": args.bin_min,
        "z0": args.z0,
        "f0": args.f0,
        "min_dur_min": args.min_dur,
        "n_stations_seen": int(ts["n_stations"].max()),
        "episodes_n": int(len(episodes)),
        "episodes_max_duration_min": float(max([e.duration_min for e in episodes], default=0.0)),
        "max_f_active": float(ts["f_active"].max()),
        "max_mean_absz": float(ts["mean_absz"].max()),
    }

    obs["p_max_f_active"] = p_value(null_df["max_f_active"], obs["max_f_active"], tail="ge")
    obs["p_n_episodes"] = p_value(null_df["n_episodes"], obs["episodes_n"], tail="ge")
    obs["p_max_episode_duration_min"] = p_value(null_df["max_episode_duration_min"], obs["episodes_max_duration_min"], tail="ge")

    out_dir = root / "resultados" / event / "robust_precursors"
    plots_dir = root / "resultados" / event / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ts_out = out_dir / "precursors_timeseries.csv"
    ts.to_csv(ts_out, index=False)

    null_out = out_dir / "null_summary.csv"
    null_df.to_csv(null_out, index=False)

    summary_out = out_dir / "robust_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump({"observed": obs, "null_percentiles": null_summary}, f, indent=2)

    out_png = out_dir / f"robust_precursors_{event}.png"
    make_figure(event, ts, episodes, null_summary, out_png)

    plots_png = plots_dir / f"robust_precursors_{event}.png"
    try:
        import shutil
        shutil.copyfile(out_png, plots_png)
    except Exception:
        pass

    print("\n=========================================")
    print(f"ROBUST PRECURSORS — {event}")
    print("-----------------------------------------")
    print(f"Mainshock (UTC): {mainshock}")
    print(f"Stations (max per bin): {obs['n_stations_seen']}")
    print(f"Episodes detected: {obs['episodes_n']}")
    print(f"Max episode duration (min): {obs['episodes_max_duration_min']:.1f}")
    print(f"Max f_active: {obs['max_f_active']:.3f}   p={obs['p_max_f_active']:.4f}")
    print(f"N episodes p-value: p={obs['p_n_episodes']:.4f}")
    print(f"Max duration p-value: p={obs['p_max_episode_duration_min']:.4f}")
    print("-----------------------------------------")
    print(f"[OK] TS CSV:   {ts_out}")
    print(f"[OK] NULL CSV: {null_out}")
    print(f"[OK] JSON:     {summary_out}")
    print(f"[OK] FIG:      {out_png}")
    print(f"[OK] FIG copy: {plots_png}")
    print("=========================================\n")


if __name__ == "__main__":
    main()

# FIX: allow running on control paths and fallback mainshock time from name