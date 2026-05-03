#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_rotation_3c.py
- Lee MSEED preprocesados 24h (3 componentes)
- Calcula rotación PCA
- Genera serie TAMC-ROT por estación
- Guarda resultados en:
    resultados/<EVENT>/rot/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import obspy
from obspy import read
import sys

# =======================
# Leer argumento EVENTO
# =======================
if len(sys.argv) < 2:
    raise SystemExit("Uso: python 01_rotation_3c.py <evento>")

EVENT = sys.argv[1]
print(f"\n>>> Ejecutando 01_rotation_3c para EVENTO = {EVENT}\n")

# =======================
# Paths
# =======================
ROOT_DIR = Path(__file__).resolve().parents[2]


def get_paths_for_event(event_name: str):
    """
    Devuelve:
      - pre_base: data/<evento>/preprocessed/mseed_24h
      - out_base: resultados/<evento>/rot
    """
    pre_base = ROOT_DIR / "data" / event_name / "preprocessed" / "mseed_24h"
    out_base = ROOT_DIR / "resultados" / event_name / "rot"
    out_base.mkdir(parents=True, exist_ok=True)
    return pre_base, out_base


# =======================
# Funciones principales
# =======================
def compute_tamc_rotation(stream: obspy.Stream):
    """
    Calcula la rotación PCA y devuelve:
      - tamc_rot (np.ndarray)
      - rotation_matrix (np.ndarray 3x3)
    """
    if len(stream) < 3:
        raise ValueError("Se requieren 3 componentes para la rotación")

    data = np.vstack([tr.data for tr in stream])
    data = data - data.mean(axis=1, keepdims=True)

    cov = np.cov(data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    rotated = eigvecs.T @ data
    tamc_rot = np.linalg.norm(rotated[:2, :], axis=0)

    return tamc_rot, eigvecs


def run_rotation_3c_for_event(event_name: str):
    pre_base, out_base = get_paths_for_event(event_name)

    print("=========================================")
    print(f"  EVENTO: {event_name}")
    print(f"  PRE:   {pre_base}")
    print(f"  OUT:   {out_base}")
    print("=========================================")

    if not pre_base.exists():
        raise FileNotFoundError(f"No existe el directorio: {pre_base}")

    files = sorted(pre_base.glob("*_pre_24h.mseed"))
    if not files:
        raise FileNotFoundError("No se encontraron archivos *_pre_24h.mseed")

    for f in files:
        print(f"\nProcesando {f.name}")
        st = read(str(f))
        st.merge(fill_value="interpolate")

        # Agrupar por estación
        stations = sorted(set(tr.stats.station for tr in st))

        for sta in stations:
            st_sta = st.select(station=sta)

            if len(st_sta) < 3:
                print(f"  [WARN] {sta}: menos de 3 componentes, se omite")
                continue

            try:
                tamc_rot, rot_matrix = compute_tamc_rotation(st_sta)
            except Exception as e:
                print(f"  [ERROR] {sta}: {e}")
                continue

            time_axis = [
                st_sta[0].stats.starttime + i * st_sta[0].stats.delta
                for i in range(len(tamc_rot))
            ]

            df = pd.DataFrame({
                "time_center_iso": [t.isoformat() for t in time_axis],
                "tamc_rot": tamc_rot
            })

            csv_out = out_base / f"tamc_24h_rot_{sta}.csv"
            npy_out = out_base / f"rotation_matrix_24h_{sta}.npy"

            df.to_csv(csv_out, index=False)
            np.save(npy_out, rot_matrix)

            print(f"  [OK] {sta}:")
            print(f"       CSV  -> {csv_out.name}")
            print(f"       NPY  -> {npy_out.name}")

    print(f"\n[OK] 01_rotation_3c completado para {event_name}\n")


# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    run_rotation_3c_for_event(event_name=EVENT)


