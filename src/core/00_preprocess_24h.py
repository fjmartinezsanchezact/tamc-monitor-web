#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
00_preprocess_24h.py
--------------------
Preprocesado 24h multiestación para un terremoto (evento).
"""

from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

from obspy import read, Stream
from obspy.signal.filter import bandpass

# =======================
# Leer argumento EVENTO
# =======================

if len(sys.argv) > 1:
    EVENT = sys.argv[1]
else:
    EVENT = "tohoku2011"

print(f"\n>>> Ejecutando script para EVENTO = {EVENT}\n")

# ---------------------------------------------------------
# Parámetros globales
# ---------------------------------------------------------

DEFAULT_EVENT = "tohoku2011"
TARGET_FS = 1.0
FREQMIN = 0.05
FREQMAX = 1.0


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_event_paths(event: str):
    base = get_project_root()
    event_base = base / "data" / event
    return {
        "base": base,
        "event_base": event_base,
        "raw_dir": event_base / "raw",
        "pre_dir": event_base / "preprocessed" / "mseed_24h",
    }


# ---------------------------------------------------------
# Escaneo de MSEED
# ---------------------------------------------------------

def scan_raw_mseed_files(raw_dir: Path):
    files_by_station = defaultdict(list)
    locations_by_station = defaultdict(set)

    mseed_files = sorted(raw_dir.rglob("*.mseed"))
    if not mseed_files:
        print(f"[WARN] No hay archivos .mseed en {raw_dir}")
        return files_by_station, locations_by_station

    print(f"  Archivos .mseed encontrados: {len(mseed_files)}")

    for f in mseed_files:
        try:
            st = read(f, headonly=True)
        except Exception as e:
            print(f"  [WARN] No se pudo leer cabecera de {f.name}: {e}")
            continue

        if not st:
            continue

        tr = st[0]
        net = (tr.stats.network or "--").strip()
        sta = (tr.stats.station or "XXXX").strip()
        loc = (tr.stats.location or "").strip()

        files_by_station[(net, sta, loc)].append(f)
        locations_by_station[(net, sta)].add(loc)

    return files_by_station, locations_by_station


def choose_location_per_station(locations_by_station):
    chosen = {}
    for (net, sta), locs in locations_by_station.items():
        chosen[(net, sta)] = "00" if "00" in locs else sorted(locs)[0]
    return chosen


# ---------------------------------------------------------
# Preprocesado por estación
# ---------------------------------------------------------

def preprocess_station(net, sta, loc, files, out_dir):
    net_sta = f"{net}.{sta}"
    print(f"\n[{net_sta} loc='{loc}'] Preprocesando 24h...")

    st = Stream()
    for f in files:
        try:
            st += read(f)
        except Exception as e:
            print(f"  [WARN] No se pudo leer {f.name}: {e}")

    st = st.select(channel="BH?")
    st = Stream(tr for tr in st if (tr.stats.location or "").strip() == loc)

    if len(st) < 3:
        print("  [WARN] Menos de 3 canales BH?, se omite.")
        return

    # ---------- FIX GLOBAL DE DTYPE ----------
    for tr in st:
        if not np.issubdtype(tr.data.dtype, np.floating):
            tr.data = tr.data.astype(np.float32)
    # ----------------------------------------

    st.merge(method=1, fill_value=0.0)
    st.sort(keys=["channel"])

    # Intersección temporal
    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    if t1 <= t0:
        print("  [WARN] No hay intersección temporal.")
        return

    st.trim(t0, t1, pad=True, fill_value=0.0)

    # Preprocesado estándar
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.01)

    for tr in st:
        tr.data = bandpass(
            tr.data,
            FREQMIN,
            FREQMAX,
            df=tr.stats.sampling_rate,
            corners=4,
            zerophase=True,
        )

        if abs(tr.stats.sampling_rate - TARGET_FS) > 1e-6:
            tr.interpolate(sampling_rate=TARGET_FS)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{net_sta}_pre_24h.mseed"
    st.write(out_path, format="MSEED")
    print(f"  [OK] Guardado {out_path}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main(event=DEFAULT_EVENT):
    print("=== 00 - Preprocesado 24h multiestación ===")
    print(f"Evento: {event}")

    paths = get_event_paths(event)
    raw_dir = paths["raw_dir"]
    pre_dir = paths["pre_dir"]

    print(f"  RAW dir: {raw_dir}")
    print(f"  OUT dir: {pre_dir}")

    if not raw_dir.exists():
        print(f"[ERROR] No existe {raw_dir}")
        return

    files_by_station, locations_by_station = scan_raw_mseed_files(raw_dir)
    if not files_by_station:
        print("[ERROR] No se encontraron estaciones.")
        return

    chosen_loc = choose_location_per_station(locations_by_station)

    estaciones = [
        (net, sta, loc, files)
        for (net, sta, loc), files in files_by_station.items()
        if chosen_loc[(net, sta)] == loc
    ]

    print(f"Estaciones a procesar: {len(estaciones)}")

    for net, sta, loc, files in estaciones:
        preprocess_station(net, sta, loc, files, pre_dir)

    print("\n00 - Preprocesado 24h completado.")


if __name__ == "__main__":
    main(EVENT)
