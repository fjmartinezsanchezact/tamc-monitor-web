#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_sync_multistation.py
Sincronía multiestación basada en |z| > umbral.

Layout nuevo:
  resultados/<EVENT>/metrics/tamc_24h_metrics_allstations.csv
  resultados/<EVENT>/sync/sync_multistation.csv
"""

import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) < 2:
    raise SystemExit("Uso: python 04_sync_multistation.py <evento>")

EVENT = sys.argv[1]
print(f"\n>>> Ejecutando script para EVENTO = {EVENT}\n")

Z_THRESHOLD = 2.5
MIN_STATIONS = 2


def main():
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / "resultados" / EVENT

    metrics_file = base / "metrics" / "tamc_24h_metrics_allstations.csv"
    out_dir = base / "sync"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 41)
    print(f"  EVENTO: {EVENT}")
    print(f"  METRICS: {metrics_file}")
    print(f"  OUT: {out_dir}")
    print("=" * 41)

    if not metrics_file.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de métricas: {metrics_file}")

    df = pd.read_csv(metrics_file)
    df["time_center_iso"] = pd.to_datetime(df["time_center_iso"], format="mixed", errors="coerce")
    df = df.dropna(subset=["time_center_iso", "zscore", "station_id"])

    estaciones = df["station_id"].unique()
    print(f"Estaciones detectadas: {list(estaciones)}")

    pivot = df.pivot(index="time_center_iso", columns="station_id", values="zscore")

    activation = pivot.abs() > Z_THRESHOLD
    activation["active_count"] = activation.sum(axis=1)
    activation["active_frac"] = activation["active_count"] / len(estaciones)
    activation["sync_flag"] = activation["active_count"] >= MIN_STATIONS

    out_file = out_dir / "sync_multistation.csv"
    activation.to_csv(out_file)

    print(f"\n[OK] Sincronía multiestación guardada en:\n   {out_file}\n")
    print("[OK] 04_sync_multistation completado.\n")


if __name__ == "__main__":
    main()

