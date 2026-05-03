#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_scan_criticality.py
Escanea la serie TAMC (zscore) para detectar picos y valles.

FIX:
- Soporta EVENT con "/" (ej: padre/mainshock o padre/control_01...)
- Usa EVENT_TAG (safe) SOLO para nombres de archivo.
"""

from pathlib import Path
import pandas as pd
import sys

if len(sys.argv) < 2:
    raise SystemExit("Uso: python 03_scan_criticality.py <evento>")

EVENT = sys.argv[1].strip()
EVENT_TAG = EVENT.replace("/", "_").replace("\\", "_")

print(f"\n>>> Ejecutando script para EVENTO = {EVENT}\n")

Z_THRESH_PEAK = 2.0
Z_THRESH_VALLEY = -2.0
MIN_SEPARATION = 5


def find_local_extrema_zscore(df, z_col="zscore", time_col="time_center_iso"):
    df = df.sort_values(time_col).reset_index(drop=True)
    z = df[z_col].values
    times = df[time_col].values

    extrema = []
    last_idx = -MIN_SEPARATION

    for i in range(1, len(z) - 1):
        if i - last_idx < MIN_SEPARATION:
            continue

        if z[i] >= z[i - 1] and z[i] >= z[i + 1] and z[i] >= Z_THRESH_PEAK:
            extrema.append((times[i], z[i], "peak", i))
            last_idx = i

        elif z[i] <= z[i - 1] and z[i] <= z[i + 1] and z[i] <= Z_THRESH_VALLEY:
            extrema.append((times[i], z[i], "valley", i))
            last_idx = i

    return pd.DataFrame(extrema, columns=[time_col, z_col, "kind", "idx_local"])


def main():
    ROOT = Path(__file__).resolve().parents[2]
    base = ROOT / "resultados" / EVENT

    metrics_file = base / "metrics" / "tamc_24h_metrics_allstations.csv"
    scan_dir = base / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)

    df = pd.read_csv(metrics_file)

    if "station_id" not in df.columns:
        raise KeyError(f"metrics_allstations no tiene 'station_id'. Columnas: {list(df.columns)}")

    all_extrema = []

    for sta in sorted(df["station_id"].unique()):
        df_sta = df[df["station_id"] == sta]
        extrema = find_local_extrema_zscore(df_sta)

        # ✅ nombre seguro (EVENT_TAG) para que no se interprete '/' como carpeta
        out_sta = scan_dir / f"scan_{EVENT_TAG}_24h_{sta}.csv"
        extrema.to_csv(out_sta, index=False)

        if not extrema.empty:
            extrema = extrema.copy()
            extrema["station_id"] = sta
            all_extrema.append(extrema)

    if all_extrema:
        df_all = pd.concat(all_extrema, ignore_index=True)
        out_all = scan_dir / f"scan_{EVENT_TAG}_24h_allstations.csv"
        df_all.to_csv(out_all, index=False)

    print("[OK] 03_scan_criticality completado.")


if __name__ == "__main__":
    main()

