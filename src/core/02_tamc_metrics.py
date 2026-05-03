#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_tamc_metrics.py  (FIX: EVENT con '/' no rompe plots)

- Calcula z-score por estación desde tamc_24h_rot_*.csv (en resultados/<EVENT>/rot/)
- Escribe:
    resultados/<EVENT>/metrics/tamc_24h_metrics_<STATION>.csv
    resultados/<EVENT>/metrics/tamc_24h_metrics_allstations.csv
- Dibuja (AHORA SEGURO en Windows aunque EVENT tenga '/'):
    resultados/<EVENT>/plots/zscore_multistation_<EVENT_TAG>.png

NUEVO:
- Si existe el CSV de extremos generado por 03_scan_criticality.py:
    resultados/<EVENT>/scan/scan_<EVENT(_TAG)>_24h_allstations.csv
  entonces genera:
    resultados/<EVENT>/plots/extremos_timeline_<EVENT_TAG>.png

Uso:
  python 02_tamc_metrics.py <evento>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Arg: EVENT
# -------------------------
if len(sys.argv) < 2:
    raise SystemExit("Uso: python 02_tamc_metrics.py <evento>")

EVENT = sys.argv[1]
EVENT_TAG = EVENT.replace("/", "_").replace("\\", "_")

print(f"\n>>> Ejecutando 02_tamc_metrics para EVENTO = {EVENT}\n")


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROT_DIR    = PROJECT_ROOT / "resultados" / EVENT / "rot"
OUT_DIR    = PROJECT_ROOT / "resultados" / EVENT / "metrics"
PLOTS_DIR  = PROJECT_ROOT / "resultados" / EVENT / "plots"
EVENTS_CSV = PROJECT_ROOT / "src" / "events.csv"

# CSV de extremos (salida de 03_scan_criticality.py)
# OJO: el filename puede venir en modo viejo (EVENT) o modo safe (EVENT_TAG).
SCAN_FILE_OLD = PROJECT_ROOT / "resultados" / EVENT / "scan" / f"scan_{EVENT}_24h_allstations.csv"
SCAN_FILE_NEW = PROJECT_ROOT / "resultados" / EVENT / "scan" / f"scan_{EVENT_TAG}_24h_allstations.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Robust datetime parsing
# -------------------------
def parse_datetime_mixed(series: pd.Series) -> pd.Series:
    """
    Parsea datetimes ISO mezcladas (con/sin microsegundos, etc.).
    Devuelve tz-aware UTC. Los errores se convierten en NaT (coerce).
    Compatible con pandas viejos y nuevos.
    """
    try:
        # pandas >= 2.0 suele soportar format="mixed"
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")
    except ValueError:
        return pd.to_datetime(series, utc=True, errors="coerce")


# -------------------------
# mainshock time (events.csv)
# -------------------------
def load_mainshock_time(event: str) -> pd.Timestamp | None:
    if not EVENTS_CSV.exists():
        return None

    df = pd.read_csv(EVENTS_CSV)
    df.columns = [c.strip() for c in df.columns]

    # columna de evento
    evcol = None
    for c in ["event", "evento", "name", "event_id"]:
        if c in df.columns:
            evcol = c
            break
    if evcol is None:
        return None

    # columna de tiempo
    tcol = None
    for c in ["origin_time", "mainshock_time", "time", "datetime", "date_time"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        for c in df.columns:
            cl = c.lower()
            if any(tok in cl for tok in ["origin", "mainshock", "time", "date", "fecha", "hora"]):
                tcol = c
                break
    if tcol is None:
        return None

    row = df[df[evcol].astype(str) == str(event)]
    if row.empty:
        return None

    t = parse_datetime_mixed(pd.Series([row.iloc[0][tcol]])).iloc[0]
    if pd.isna(t):
        return None
    return t


# -------------------------
# Plot combined z-score
# -------------------------
def plot_combined_zscore(all_metrics_csv: Path, evento: str, out_dir: Path) -> None:
    df = pd.read_csv(all_metrics_csv)

    if "time_center_iso" not in df.columns:
        raise KeyError("Falta columna 'time_center_iso' en metrics_allstations.csv")
    if "station_id" not in df.columns:
        raise KeyError("Falta columna 'station_id' en metrics_allstations.csv")
    if "zscore" not in df.columns:
        if "z_score" in df.columns:
            df["zscore"] = df["z_score"]
        else:
            raise KeyError("Falta columna 'zscore' en metrics_allstations.csv")

    df["time_dt"] = parse_datetime_mixed(df["time_center_iso"])
    df = df.dropna(subset=["time_dt"])

    mainshock_time = load_mainshock_time(evento)

    if mainshock_time is not None:
        df["t_rel_h"] = (df["time_dt"] - mainshock_time).dt.total_seconds() / 3600.0
        x_col = "t_rel_h"
        x_label = "Horas respecto al evento (t = 0 mainshock)"
    else:
        x_col = "time_dt"
        x_label = "Tiempo (UTC)"

    pivot = df.pivot(index=x_col, columns="station_id", values="zscore").sort_index()

    fig, ax = plt.subplots(figsize=(13, 6))

    for station in pivot.columns:
        ax.plot(pivot.index, pivot[station], linewidth=1.1, label=str(station))

    ax.axhline(3, linestyle="--", linewidth=1)
    ax.axhline(-3, linestyle="--", linewidth=1)
    if mainshock_time is not None:
        ax.axvline(0, linestyle="--", linewidth=1, color="black")

    ax.set_title(f"Z-score TAMC-ROT 24h — {evento}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    fig.tight_layout()

    evento_tag = evento.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"zscore_multistation_{evento_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[OK] Figura z-score multistación guardada en:\n   {out_path}")


# -------------------------
# Evolución temporal de picos/valles (conteo por bin)
# -------------------------
def plot_extrema_timeline(scan_csv: Path, evento: str, out_dir: Path, bin_minutes: int = 30) -> None:
    """
    Lee scan_<EVENT>_24h_allstations.csv (salida de 03_scan_criticality.py)
    y genera un gráfico SIMPLE (solo 2 líneas):
      - Nº de picos por bin temporal
      - Nº de valles por bin temporal
    """
    if not scan_csv.exists():
        print(f"[INFO] No existe {scan_csv}. Si quieres este plot, ejecuta 03_scan_criticality.py primero.")
        return

    df = pd.read_csv(scan_csv)

    if "time_center_iso" not in df.columns or "kind" not in df.columns:
        print(f"[WARN] {scan_csv.name} no tiene columnas esperadas. Columnas: {list(df.columns)}")
        return

    df["time"] = parse_datetime_mixed(df["time_center_iso"])
    df = df.dropna(subset=["time"]).copy()
    if df.empty:
        print(f"[INFO] {scan_csv.name} está vacío tras parsear tiempos.")
        return

    df["bin_start"] = df["time"].dt.floor(f"{int(bin_minutes)}min")
    grp = df.groupby(["bin_start", "kind"]).size().unstack(fill_value=0)

    if "peak" not in grp.columns:
        grp["peak"] = 0
    if "valley" not in grp.columns:
        grp["valley"] = 0

    grp = grp.sort_index()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(grp.index, grp["peak"], linewidth=1.8, label="Picos (peak)")
    ax.plot(grp.index, grp["valley"], linewidth=1.8, label="Valles (valley)")

    ax.set_title(f"Evolución de picos y valles (conteo / {int(bin_minutes)} min) — {evento}")
    ax.set_xlabel("Tiempo (UTC)")
    ax.set_ylabel("Nº de extremos")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    evento_tag = evento.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"extremos_timeline_{evento_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[OK] Figura evolución picos/valles guardada en:\n   {out_path}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    print("=========================================")
    print(f"  EVENTO: {EVENT}")
    print(f"  ROT_DIR: {ROT_DIR}")
    print(f"  OUT_DIR: {OUT_DIR}")
    print("=========================================")

    rot_files = sorted(ROT_DIR.glob("tamc_24h_rot_*.csv"))
    if not rot_files:
        raise FileNotFoundError(f"No se encontraron CSV en {ROT_DIR}")

    all_dfs = []

    for csv_path in rot_files:
        station_id = csv_path.stem.replace("tamc_24h_rot_", "")
        print(f"=== Estación {station_id} ===")

        df = pd.read_csv(csv_path)

        if "tamc_rot" not in df.columns:
            raise KeyError(f"Falta columna 'tamc_rot' en {csv_path.name}")
        if "time_center_iso" not in df.columns:
            raise KeyError(f"Falta columna 'time_center_iso' en {csv_path.name}")

        mean = float(df["tamc_rot"].mean())
        std  = float(df["tamc_rot"].std())

        if std == 0 or not np.isfinite(std):
            df["zscore"] = 0.0
        else:
            df["zscore"] = (df["tamc_rot"] - mean) / std

        df["station_id"] = station_id

        out_csv = OUT_DIR / f"tamc_24h_metrics_{station_id}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  [OK] Guardado: {out_csv}")

        all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_out = OUT_DIR / "tamc_24h_metrics_allstations.csv"
    all_df.to_csv(all_out, index=False)

    print(f"\n[OK] CSV combinado guardado en:\n   {all_out}\n")

    # Plot z-score multistación (con filename seguro)
    plot_combined_zscore(all_out, EVENT, PLOTS_DIR)

    # Timeline de extremos (busca archivo viejo o nuevo)
    scan_file = SCAN_FILE_OLD if SCAN_FILE_OLD.exists() else SCAN_FILE_NEW
    plot_extrema_timeline(scan_file, EVENT, PLOTS_DIR, bin_minutes=30)

    print(f"[OK] 02_tamc_metrics completado para evento {EVENT}")


if __name__ == "__main__":
    main()
