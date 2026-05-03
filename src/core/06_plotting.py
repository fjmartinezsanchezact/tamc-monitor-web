#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_plotting.py — FIXED para:
- EVENT con subcarpetas: <evento_base>/mainshock o <evento_base>/control_XX...
- Nombres de archivos seguros en Windows (EVENT_TAG, sin '/')
- t=0 (mainshock) desde events.csv usando EVENT_BASE; si no existe, fallback al timestamp del nombre

Genera (si existen inputs):
1) extremos_{EVENT_TAG}.png               (requiere scan)
2) sync_multistation_{EVENT_TAG}.png      (requiere sync)
3) zscore_multistation_{EVENT_TAG}.png    (requiere metrics)
4) precursors_timeline_{EVENT_TAG}.png    (requiere metrics)

Guarda todo en:
  resultados/<EVENT>/plots/
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =======================
# EVENTO
# =======================
if len(sys.argv) < 2:
    raise SystemExit("Uso: python 06_plotting.py <evento>")

EVENT = sys.argv[1].strip()
EVENT_TAG = EVENT.replace("/", "_").replace("\\", "_")
EVENT_BASE = EVENT.split("/")[0].split("\\")[0]  # para buscar t=0

print(f"\n>>> Generando figuras (tiempo relativo) para EVENTO = {EVENT}")
print(f"    EVENT_BASE = {EVENT_BASE}")
print(f"    EVENT_TAG  = {EVENT_TAG}\n")


# =======================
# PATHS
# =======================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "resultados" / EVENT

METRICS_FILE = BASE / "metrics" / "tamc_24h_metrics_allstations.csv"

# scan puede venir con nombre viejo o nuevo (según tu 03 corregido)
SCAN_FILE_OLD = BASE / "scan" / f"scan_{EVENT}_24h_allstations.csv"
SCAN_FILE_NEW = BASE / "scan" / f"scan_{EVENT_TAG}_24h_allstations.csv"

SYNC_FILE = BASE / "sync" / "sync_multistation.csv"

OUT_DIR = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_CSV = PROJECT_ROOT / "src" / "events.csv"


# =======================
# UTIL
# =======================
def parse_datetime_mixed(series: pd.Series) -> pd.Series:
    """Datetime robusto (pandas viejo/nuevo)."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")
    except ValueError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def parse_mainshock_from_event_name(event_base: str) -> pd.Timestamp:
    """
    Fallback: extrae YYYYMMDD_HHMMSS desde el nombre del evento base.
    Ej: ..._20170908_044919 -> 2017-09-08 04:49:19 UTC
    """
    m = re.search(r"(\d{8})_(\d{6})", event_base)
    if not m:
        raise ValueError(
            f"No pude extraer YYYYMMDD_HHMMSS desde '{event_base}'. "
            "Necesito events.csv correcto o nombre con fecha."
        )
    ymd, hms = m.group(1), m.group(2)
    s = f"{ymd}{hms}"
    t = pd.to_datetime(s, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"No pude parsear timestamp '{s}' desde '{event_base}'")
    return t


def find_event_column(df: pd.DataFrame) -> str:
    for c in ["event", "evento", "name", "event_id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def find_time_column(df: pd.DataFrame) -> str:
    for c in [
        "origin_time", "origin_time_utc",
        "mainshock_time", "mainshock_time_utc",
        "time", "time_utc", "datetime", "date_time",
        "fecha", "hora", "fecha_hora"
    ]:
        if c in df.columns:
            return c
    # fallback
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in ["origin", "mainshock", "time", "date", "fecha", "hora"]):
            return c
    raise KeyError(f"No encuentro columna de tiempo en events.csv. Columnas: {list(df.columns)}")


def load_mainshock_time(event_base: str) -> pd.Timestamp:
    """
    1) Intenta leer t=0 desde src/events.csv buscando por event_base
    2) Si no está, fallback al timestamp del nombre de carpeta
    """
    if EVENTS_CSV.exists():
        df = pd.read_csv(EVENTS_CSV)
        df.columns = [c.strip() for c in df.columns]
        evcol = find_event_column(df)
        tcol = find_time_column(df)

        row = df[df[evcol].astype(str) == str(event_base)]
        if not row.empty:
            t = pd.to_datetime(row.iloc[0][tcol], utc=True, errors="coerce")
            if not pd.isna(t):
                return t

    return parse_mainshock_from_event_name(event_base)


def load_csv_with_rel_h(path: Path, mainshock_time_utc: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normaliza columna tiempo (algunos CSV vienen con índice guardado)
    if "time_center_iso" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "time_center_iso"})
        else:
            raise KeyError(f"{path.name}: no encuentro 'time_center_iso'. Columnas: {list(df.columns)}")

    df["time_center_iso"] = parse_datetime_mixed(df["time_center_iso"])
    df = df.dropna(subset=["time_center_iso"]).copy()

    df["rel_h"] = (df["time_center_iso"] - mainshock_time_utc).dt.total_seconds() / 3600.0
    return df.sort_values("rel_h")


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def should_use_log_positive(y, ratio=30.0, min_max=20.0):
    y = _finite(y)
    if y.size == 0:
        return False
    y = y[y > 0]
    if y.size == 0:
        return False
    p95 = np.percentile(y, 95)
    mx = np.max(y)
    if p95 <= 0:
        return False
    return (mx > min_max) and ((mx / p95) > ratio)


def apply_log_positive(ax, y, label_suffix=" (log)"):
    y = _finite(y)
    y = y[y > 0]
    if y.size == 0:
        return
    eps = max(1e-3, np.percentile(y, 1) * 0.5)
    ax.set_yscale("log")
    ax.set_ylim(eps, np.max(y) * 1.2)
    ax.set_ylabel(ax.get_ylabel() + label_suffix)


def should_use_symlog_signed(y, ratio=30.0, min_max=20.0):
    y = _finite(y)
    if y.size == 0:
        return False
    a = np.abs(y)
    a = a[a > 0]
    if a.size == 0:
        return False
    p95 = np.percentile(a, 95)
    mx = np.max(a)
    if p95 <= 0:
        return False
    return (mx > min_max) and ((mx / p95) > ratio)


def apply_symlog(ax, y, linthresh=1.0, label_suffix=" (symlog)"):
    y = _finite(y)
    if y.size == 0:
        return
    a = np.abs(y)
    a = a[a > 0]
    if a.size == 0:
        return
    mx = np.max(np.abs(y))
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(-mx * 1.2, mx * 1.2)
    ax.set_ylabel(ax.get_ylabel() + label_suffix)


# =======================
# 1) EXTREMOS
# =======================
def plot_extremos(df: pd.DataFrame):
    if "kind" not in df.columns or "zscore" not in df.columns:
        print(f"[WARN] scan: faltan columnas esperadas. Columnas: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    peaks = df[df["kind"] == "peak"]
    valleys = df[df["kind"] == "valley"]

    ax.scatter(peaks["rel_h"], peaks["zscore"], s=20, label="Peaks")
    ax.scatter(valleys["rel_h"], valleys["zscore"], s=20, label="Valleys")

    ax.axvline(0, color="red", linestyle="--", linewidth=2)

    ax.set_title(f"Extremos TAMC — {EVENT}")
    ax.set_xlabel("Horas respecto al evento (t = 0)")
    ax.set_ylabel("Z-score")
    ax.legend()
    ax.grid(True)

    y_all = np.r_[peaks["zscore"].to_numpy(), valleys["zscore"].to_numpy()]
    if should_use_log_positive(y_all, ratio=30.0, min_max=20.0):
        apply_log_positive(ax, y_all)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"extremos_{EVENT_TAG}.png", dpi=200)
    plt.close(fig)


# =======================
# 2) SINCRONÍA
# =======================
def plot_sync(df: pd.DataFrame):
    if "active_frac" not in df.columns:
        print(f"[WARN] sync: falta 'active_frac'. Columnas: {list(df.columns)}")
        return

    # sync_flag puede no existir -> lo hacemos opcional
    if "sync_flag" not in df.columns:
        df = df.copy()
        df["sync_flag"] = False

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["rel_h"], df["active_frac"], label="active_frac")

    sync = df[df["sync_flag"].astype(bool)]
    if not sync.empty:
        ax.scatter(sync["rel_h"], sync["active_frac"], label="sync_flag")

    ax.axvline(0, color="red", linestyle="--", linewidth=2)

    ax.set_title(f"Sincronía multiestación — {EVENT}")
    ax.set_xlabel("Horas respecto al evento (t = 0)")
    ax.set_ylabel("Fracción estaciones activas")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"sync_multistation_{EVENT_TAG}.png", dpi=200)
    plt.close(fig)


# =======================
# 3) Z-SCORE MULTIESTACIÓN
# =======================
def plot_zscore_multi(df: pd.DataFrame):
    if "station_id" not in df.columns or "zscore" not in df.columns:
        print(f"[WARN] metrics: faltan columnas esperadas. Columnas: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    all_z = []
    for sta in sorted(df["station_id"].unique()):
        sub = df[df["station_id"] == sta]
        z = sub["zscore"].to_numpy(dtype=float)
        all_z.append(z)
        ax.plot(sub["rel_h"], z, linewidth=1, label=str(sta))

    all_z = np.concatenate(all_z) if len(all_z) else np.array([])

    ax.axhline(3, color="red", linestyle="--")
    ax.axhline(-3, color="red", linestyle="--")
    ax.axvline(0, color="red", linestyle="--")

    ax.set_title(f"Z-score TAMC-ROT 24h — {EVENT}")
    ax.set_xlabel("Horas respecto al evento (t = 0)")
    ax.set_ylabel("Z-score")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    if should_use_symlog_signed(all_z, ratio=30.0, min_max=20.0):
        apply_symlog(ax, all_z, linthresh=1.0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"zscore_multistation_{EVENT_TAG}.png", dpi=200)
    plt.close(fig)


# =======================
# 4) PRECURSORES
# =======================
def plot_precursors(df: pd.DataFrame):
    if "zscore" not in df.columns:
        print(f"[WARN] metrics: falta 'zscore'. Columnas: {list(df.columns)}")
        return

    df = df.copy()
    df["abs_z"] = df["zscore"].astype(float).abs()

    grp = df.groupby("rel_h")
    m_abs = grp["abs_z"].mean()
    chi = grp["zscore"].var()

    smooth = m_abs.rolling(25, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(m_abs.index, smooth, label="|z| medio (suavizado)")
    ax1.axvline(0, color="red", linestyle="--")

    ax1.set_xlabel("Horas respecto al evento (t = 0)")
    ax1.set_ylabel("|z| medio")
    ax1.grid(True)

    if should_use_log_positive(smooth.to_numpy(), ratio=30.0, min_max=10.0):
        apply_log_positive(ax1, smooth.to_numpy(), label_suffix=" (log)")

    ax2 = ax1.twinx()
    ax2.plot(chi.index, chi, alpha=0.7, label="χ(t)")
    ax2.set_ylabel("Susceptibilidad")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"precursors_timeline_{EVENT_TAG}.png", dpi=200)
    plt.close(fig)


# =======================
# MAIN
# =======================
def main():
    # t=0 (mainshock) válido para mainshock y para controles
    mainshock_time_utc = load_mainshock_time(EVENT_BASE)
    print(f"[06] mainshock_utc = {mainshock_time_utc}\n")

    # metrics es lo mínimo para generar 2 plots
    if not METRICS_FILE.exists():
        raise FileNotFoundError(METRICS_FILE)

    df_metrics = load_csv_with_rel_h(METRICS_FILE, mainshock_time_utc)

    # scan opcional
    scan_path = SCAN_FILE_NEW if SCAN_FILE_NEW.exists() else SCAN_FILE_OLD
    df_scan = None
    if scan_path.exists():
        df_scan = load_csv_with_rel_h(scan_path, mainshock_time_utc)
    else:
        print(f"[INFO] No existe scan_allstations (03 no ejecutado). Probé:\n  - {SCAN_FILE_NEW}\n  - {SCAN_FILE_OLD}\n")

    # sync opcional
    df_sync = None
    if SYNC_FILE.exists():
        df_sync = load_csv_with_rel_h(SYNC_FILE, mainshock_time_utc)
    else:
        print(f"[INFO] No existe {SYNC_FILE} (04 no ejecutado).\n")

    # plots
    if df_scan is not None:
        plot_extremos(df_scan)
    plot_zscore_multi(df_metrics)
    plot_precursors(df_metrics)
    if df_sync is not None:
        plot_sync(df_sync)

    print(f"[OK] Figuras guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
