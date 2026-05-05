#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAMC Master Menu (Bilingual ES/EN)
=================================

ES:
- Orquesta descarga (data/), procesamiento (00..07) y tests (10/11/09-block).
- IMPORTANTE: NO abre PNGs en pantalla (evita ventanas emergentes durante el pipeline).
EN:
- Orchestrates download (data/), processing (00..07) and tests (10/11/09-block).
- IMPORTANT: Does NOT open PNGs on screen (prevents pop-up windows during pipeline).

Opciones / Options:
1) Descargar / Monitorizar (Download / Monitor)
2) Procesar data/ existentes (Process existing data/)
3) RunTests sobre resultados/ (incluye resultados/otros/*)
4) Análisis inter-eventos (13..18) / Inter-event analysis (13..18)
5) Clasificación de terremotos (19) / Earthquake clustering (19)
0) Salir / Exit
"""

from __future__ import annotations

import os
import sys
import json
import random
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client, RoutingClient
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics.base import gps2dist_azimuth

# --- Plot summary (RECENT) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Shared subprocess helper (needed by menu option 5 as well as other menus)
# ---------------------------------------------------------------------------
def run_cmd(cmd, cwd=None, env=None):
    """Run a command, stream stdout/stderr, and return the process return code."""
    import subprocess
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    except FileNotFoundError as e:
        print(f"[ERROR] No se pudo ejecutar: {e}")
        return 127
    if p.stdout:
        print(p.stdout.rstrip())
    if p.stderr:
        # stderr a menudo trae warnings útiles; lo mostramos igual
        print(p.stderr.rstrip())
    return int(p.returncode)



# =========================
# PLOTS / VENTANAS
# =========================
# ES: NO abrir PNGs en pantalla (evita ventanas).
# EN: Do NOT open PNGs on screen (prevents popups).
AUTO_OPEN_PLOTS = False


# -------------------------
# Config
# -------------------------
CFG = dict(
    usgs_limit=200,
    usgs_minmag=7.0,
    window_hours=24.0,
    window_centered=True,  # True: t0±(window_hours/2); False: [t0-window_hours, t0]


    control_days=10,
    exclusion_days_before=30,
    max_lookback_days=365,
    include_post_event_controls=True,

    minradius_deg=0.0,
    maxradius_deg=30.0,

    station_provider="IRIS",
    data_provider="IRIS",

    net="*",
    sta="*",
    loc="*",
    chanpat="BH?",

    min_coverage=0.90,

    n_nearest=5,
    unique_by_net_sta=True,

    random_seed=7,

    default_windows_base=r"C:\Users\PC\Desktop\tamcsismico\data",
)

USGS_EVENT_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"


# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def re_safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:160]

def log(msg: str, logpath: str) -> None:
    print(msg, flush=True)
    with open(logpath, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def get_base_dir() -> str:
    env = os.environ.get("TAMC_BASE")
    if env:
        return env
    if os.path.isdir(CFG["default_windows_base"]):
        return CFG["default_windows_base"]
    return os.path.join(os.getcwd(), "tamcsismico", "data")

def build_data_client():
    try:
        c = Client(CFG.get("data_provider", "IRIS"))
        if getattr(c, "services", None) and c.services.get("dataselect"):
            return c
    except Exception:
        pass
    return RoutingClient("iris-federator")

def pct_coverage(st: Stream, start: UTCDateTime, end: UTCDateTime) -> float:
    if len(st) == 0:
        return 0.0
    st2 = st.copy().merge(method=1, fill_value=None)
    duration = float(end - start)
    if duration <= 0:
        return 0.0
    gap_sum = 0.0
    for g in st2.get_gaps():
        gap_sum += float(g[6])
    covered = max(0.0, duration - gap_sum)
    return max(0.0, min(1.0, covered / duration))

def fetch_top_earthquakes_last_20y(limit: int, minmag: float) -> List[dict]:
    now = UTCDateTime()
    start = now - 20 * 365.25 * 86400
    params = dict(
        format="geojson",
        eventtype="earthquake",
        starttime=start.strftime("%Y-%m-%d"),
        endtime=now.strftime("%Y-%m-%d"),
        orderby="magnitude",
        limit=str(limit),
        minmagnitude=str(minmag),
    )
    r = requests.get(USGS_EVENT_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("features", [])


def fetch_usgs_event_by_id(eventid: str) -> Optional[dict]:
    """
    ES: Descarga un único evento USGS en formato geojson usando eventid.
    EN: Fetch a single USGS event (geojson) by eventid.
    """
    params = dict(format="geojson", eventid=str(eventid))
    try:
        r = requests.get(USGS_EVENT_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        # USGS devuelve un "Feature" directo para eventid
        if isinstance(j, dict) and j.get("type") == "Feature":
            return j
    except Exception as e:
        print(f"[ERROR] No pude descargar evento USGS id={eventid}: {repr(e)}")
    return None

def pretty_event_row(i: int, feat: dict) -> str:
    p = feat["properties"]
    g = feat["geometry"]
    t = UTCDateTime(p["time"] / 1000.0)
    lon, lat, depth_km = g["coordinates"]
    return (
        f"{i:02d}) Mw {p['mag']:>4} | "
        f"{t.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
        f"depth {depth_km:>6} km | {p['place']} | id={feat.get('id','')}"
    )

def parse_indices(raw: str, nmax: int) -> List[int]:
    parts = [p for p in raw.replace(",", " ").split() if p]
    out: List[int] = []
    for p in parts:
        if not p.isdigit():
            return []
        k = int(p)
        if not (1 <= k <= nmax):
            return []
        out.append(k)
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def infer_event_date_from_name(ev_name: str) -> Optional[str]:
    """
    Intenta inferir fecha YYYY-MM-DD desde el nombre del evento.
    Busca patrones tipo _YYYYMMDD_ o _YYYYMMDD al final.
    """
    m = re.search(r"(?:^|_)(\d{8})(?:_|$)", ev_name)
    if not m:
        return None
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

def ensure_event_root_metrics_for_placebo(event_key: str) -> None:
    """
    Algunos scripts (p.ej. 10_placebo_matched_controls_FINAL.py) esperan:
      resultados/<EVENTO>/metrics/tamc_24h_metrics_allstations.csv

    Pero nuestro pipeline genera:
      resultados/<EVENTO>/mainshock/metrics/tamc_24h_metrics_allstations.csv

    Esta función copia (si falta) el CSV desde mainshock/metrics hacia metrics/.
    """
    pr = find_project_root()
    resultados_dir = pr / "resultados"

    ev_dir = resultados_dir / Path(event_key)
    # soporta "otros/<EVENTO>"
    if event_key.startswith("otros/"):
        ev_dir = resultados_dir / Path(event_key)

    src_csv = ev_dir / "mainshock" / "metrics" / "tamc_24h_metrics_allstations.csv"
    dst_dir = ev_dir / "metrics"
    dst_csv = dst_dir / "tamc_24h_metrics_allstations.csv"

    if dst_csv.exists():
        return
    if src_csv.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_csv, dst_csv)
        print(f"[INFO] Copié metrics para placebo: {src_csv} -> {dst_csv}")


def choose_control_days(event_end: UTCDateTime) -> List[UTCDateTime]:
    random.seed(int(CFG["random_seed"]))
    exclusion = int(CFG["exclusion_days_before"])
    max_lookback = int(CFG["max_lookback_days"])
    include_post = bool(CFG["include_post_event_controls"])

    cand: List[UTCDateTime] = []
    for d in range(exclusion, max_lookback + 1):
        cand.append(event_end - d * 86400)

    if include_post:
        now = UTCDateTime()
        n_post = int((now - event_end) // 86400)
        for d in range(1, n_post + 1):
            cand.append(event_end + d * 86400)

    random.shuffle(cand)
    return cand




# -------------------------
# Manual preset helper
# -------------------------
def make_manual_feature(place: str, time_utc_iso: str, lat: float, lon: float, depth_km: float = 0.0, mag: float = 0.0) -> dict:
    """Create a USGS-like GeoJSON feature for presets when no USGS eventid exists."""
    t = UTCDateTime(time_utc_iso)
    return {
        "type": "Feature",
        "properties": {
            "mag": float(mag),
            "place": place,
            "time": int(t.timestamp * 1000),
            "title": f"M{mag:.1f} - {place}",
        },
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat), float(depth_km)]},
        "id": "manual",
    }

# -------------------------
# TauP helpers (phase times)
# -------------------------
def estimate_p_arrival_hours(dist_km: float, depth_km: float) -> Optional[float]:
    """Return predicted P travel time (hours) using iasp91. Returns None on failure."""
    try:
        model = TauPyModel(model="iasp91")
        dist_deg = float(dist_km) / 111.19
        arrs = model.get_travel_times(source_depth_in_km=float(depth_km), distance_in_degree=dist_deg, phase_list=["P", "pP", "Pn"])
        if not arrs:
            return None
        # pick earliest
        tsec = min(a.time for a in arrs)
        return float(tsec) / 3600.0
    except Exception:
        return None

def write_p_arrival_metadata(out_dir: str, stations: List['StationCand'], depth_km: float) -> None:
    """Compute per-station P travel-time estimates and save JSON + print summary."""
    try:
        vals = []
        per = []
        for s in stations:
            h = estimate_p_arrival_hours(s.dist_km, depth_km)
            if h is None:
                continue
            vals.append(h)
            per.append({"net": s.net, "sta": s.sta, "loc": s.loc, "dist_km": s.dist_km, "p_hours": h})
        if not vals:
            return
        vals_sorted = sorted(vals)
        med = vals_sorted[len(vals_sorted)//2]
        mn = vals_sorted[0]
        mx = vals_sorted[-1]
        payload = {
            "phase": "P (earliest of P/pP/Pn)",
            "model": "iasp91",
            "depth_km": float(depth_km),
            "summary": {"min_hours": mn, "median_hours": med, "max_hours": mx, "n": len(vals_sorted)},
            "per_station": per,
        }
        jpath = os.path.join(out_dir, "p_arrival_meta.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[TAUP] P arrival estimate saved: {jpath}")
        print(f"[TAUP] P arrival hours (min/median/max over stations): {mn:.3f} / {med:.3f} / {mx:.3f} h")
    except Exception:
        return

# -------------------------
# Station selection
# -------------------------
@dataclass(frozen=True)
class StationKey:
    net: str
    sta: str
    loc: str

@dataclass
class StationCand:
    net: str
    sta: str
    loc: str
    stalat: float
    stalon: float
    dist_km: float

def fetch_candidate_stations(
    station_client: Client,
    start: UTCDateTime,
    end: UTCDateTime,
    ev_lat: float,
    ev_lon: float,
) -> List[StationCand]:
    inv = station_client.get_stations(
        network=CFG["net"],
        station=CFG["sta"],
        location=CFG["loc"],
        channel=CFG["chanpat"],
        starttime=start,
        endtime=end,
        latitude=ev_lat,
        longitude=ev_lon,
        minradius=CFG["minradius_deg"],
        maxradius=CFG["maxradius_deg"],
        level="channel",
    )

    cands: List[StationCand] = []
    for net in inv:
        for sta in net.stations:
            if sta.latitude is None or sta.longitude is None:
                continue
            dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, sta.latitude, sta.longitude)
            dist_km = float(dist_m) / 1000.0

            locs = set()
            for ch in sta.channels:
                if ch.code.startswith("BH"):
                    locs.add(ch.location_code or "")

            for loc in sorted(locs):
                cands.append(StationCand(net.code, sta.code, loc, float(sta.latitude), float(sta.longitude), dist_km))

    if bool(CFG["unique_by_net_sta"]):
        best: Dict[Tuple[str, str], StationCand] = {}
        for c in cands:
            k = (c.net, c.sta)
            if k not in best or c.dist_km < best[k].dist_km:
                best[k] = c
        cands = list(best.values())

    cands.sort(key=lambda x: x.dist_km)
    return cands


# -------------------------
# Download
# -------------------------
def download_station_window(
    data_client,
    start: UTCDateTime,
    end: UTCDateTime,
    sta: StationCand,
    out_raw_dir: str,
    logpath: str,
    tag: str,
) -> bool:
    try:
        st = data_client.get_waveforms(
            network=sta.net,
            station=sta.sta,
            location=(sta.loc if sta.loc != "" else "--"),
            channel=CFG["chanpat"],
            starttime=start,
            endtime=end,
            attach_response=False,
        )
        cov = pct_coverage(st, start, end)
        if cov < float(CFG["min_coverage"]) or len(st) == 0:
            log(f"[{tag}] DROP {sta.net}.{sta.sta}.{sta.loc} dist_km={sta.dist_km:.1f} cov={cov:.3f}", logpath)
            return False

        chans = sorted({tr.stats.channel for tr in st})
        chan_tag = "-".join(chans) if chans else "BH"
        fname = f"{sta.net}.{sta.sta}.{sta.loc}.{chan_tag}_{start.date}_{tag}.mseed".replace("..", ".")
        fpath = os.path.join(out_raw_dir, fname)
        st.write(fpath, format="MSEED")

        log(f"[{tag}] OK   {sta.net}.{sta.sta}.{sta.loc} dist_km={sta.dist_km:.1f} cov={cov:.3f} traces={len(st)}", logpath)
        return True

    except FDSNNoDataException:
        log(f"[{tag}] NODATA {sta.net}.{sta.sta}.{sta.loc} dist_km={sta.dist_km:.1f}", logpath)
        return False
    except Exception as e:
        log(f"[{tag}] ERROR {sta.net}.{sta.sta}.{sta.loc} dist_km={sta.dist_km:.1f} err={repr(e)}", logpath)
        return False


def run_event_download(feat: dict, control_days: Optional[int] = None, base_dir_override: Optional[str] = None) -> int:
    p = feat["properties"]
    g = feat["geometry"]

    eid = feat.get("id", "event")
    mag = p.get("mag", None)
    place = p.get("place", "Unknown place")
    t_ms = p.get("time", None)
    if not t_ms:
        print(f"[SKIP] {eid}: missing time", flush=True)
        return 2

    event_time = UTCDateTime(t_ms / 1000.0)
    lon, lat, depth_km = g["coordinates"]
    ev_lat = float(lat)
    ev_lon = float(lon)

    window_hours = float(CFG["window_hours"])
    centered = bool(CFG.get("window_centered", True))
    if centered:
        half = 0.5 * window_hours * 3600
        event_start = event_time - half
        event_end = event_time + half
    else:
        event_start = event_time - window_hours * 3600
        event_end = event_time


    base_dir = base_dir_override or get_base_dir()
    ensure_dir(base_dir)

    event_stamp = event_time.strftime("%Y%m%d_%H%M%S")
    mag_str = f"{mag:.1f}" if isinstance(mag, (int, float)) else str(mag)
    root_name = re_safe(f"{place}_M{mag_str}_{event_stamp}")
    root_dir = os.path.join(base_dir, root_name)
    ensure_dir(root_dir)

    mainshock_dir = os.path.join(root_dir, "mainshock")
    mainshock_raw = os.path.join(mainshock_dir, "raw")
    ensure_dir(mainshock_raw)

    candidates = choose_control_days(event_end)
    need = int(control_days) if control_days is not None else int(CFG["control_days"])

    logpath = os.path.join(root_dir, "runlog.txt")
    with open(logpath, "w", encoding="utf-8") as f:
        f.write("RUNLOG\n")
        f.write(f"USGS_EVENT_ID={eid}\n")
        f.write(f"MAG={mag_str}\n")
        f.write(f"PLACE={place}\n")
        f.write(f"LAT={ev_lat}\nLON={ev_lon}\nDEPTH_KM={depth_km}\n")
        f.write(f"EVENT_START={event_start.isoformat()}\nEVENT_END={event_end.isoformat()}\n")
        f.write(f"CONTROL_DAYS={need}\n")
        f.write(f"CFG={json.dumps(CFG, ensure_ascii=False)}\n\n")

    log(f"\n=== EVENT {eid} ===", logpath)
    log(f"{place} | Mw {mag_str} | {event_time.isoformat()}Z", logpath)
    log(f"Root: {root_dir}", logpath)

    station_client = Client(CFG["station_provider"])
    data_client = build_data_client()

    cands = fetch_candidate_stations(station_client, event_start, event_end, ev_lat, ev_lon)
    log(f"Candidates found: {len(cands)} (sorted by distance)", logpath)

    nearest_ok: List[StationCand] = []
    for sta in cands:
        if len(nearest_ok) >= int(CFG["n_nearest"]):
            break
        ok = download_station_window(data_client, event_start, event_end, sta, mainshock_raw, logpath, tag="EVENT")
        if ok:
            nearest_ok.append(sta)

    # Save TauP-based P arrival estimates for interpretation (optional)
    write_p_arrival_metadata(mainshock_dir, nearest_ok, depth_km)

    if len(nearest_ok) < int(CFG["n_nearest"]):
        log(f"[FAIL] Only {len(nearest_ok)}/{CFG['n_nearest']} stations passed QC for EVENT. Aborting this event.", logpath)
        return 2

    fixed_path = os.path.join(root_dir, "fixed_stations.json")
    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump([sta.__dict__ for sta in nearest_ok], f, indent=2, ensure_ascii=False)
    log(f"Fixed stations saved: {fixed_path}", logpath)

    got = 0
    centered = bool(CFG.get('window_centered', True))
    if centered:
        dt_start = -0.5 * window_hours * 3600
        dt_end = +0.5 * window_hours * 3600
    else:
        dt_start = -window_hours * 3600
        dt_end = 0.0

    for day_end in candidates:
        if got >= need:
            break

        ctrl_start = day_end + dt_start
        ctrl_end = day_end + dt_end

        ctrl_stamp = day_end.strftime("%Y%m%d_%H%M%S")
        ctrl_folder = f"control_{got+1:02d}_{ctrl_stamp}"
        ctrl_dir = os.path.join(root_dir, ctrl_folder)
        ctrl_raw = os.path.join(ctrl_dir, "raw")
        ensure_dir(ctrl_raw)

        tag = f"CTRL_{ctrl_stamp}"
        log(f"\n[{tag}] window {ctrl_start.isoformat()} -> {ctrl_end.isoformat()}", logpath)

        ok_count = 0
        for sta in nearest_ok:
            ok = download_station_window(data_client, ctrl_start, ctrl_end, sta, ctrl_raw, logpath, tag=tag)
            if ok:
                ok_count += 1

        if ok_count == len(nearest_ok):
            got += 1
            with open(os.path.join(ctrl_dir, "window.txt"), "w", encoding="utf-8") as f:
                f.write(f"{ctrl_start.isoformat()} -> {ctrl_end.isoformat()}\n")
            log(f"[{tag}] ACCEPT {ctrl_folder}: {ok_count}/{len(nearest_ok)}", logpath)
        else:
            log(f"[{tag}] REJECT {ctrl_folder}: {ok_count}/{len(nearest_ok)} (cleaning)", logpath)
            try:
                for fn in os.listdir(ctrl_raw):
                    try:
                        os.remove(os.path.join(ctrl_raw, fn))
                    except Exception:
                        pass
                try:
                    os.rmdir(ctrl_raw)
                    os.rmdir(ctrl_dir)
                except Exception:
                    pass
            except Exception:
                pass

    if got < need:
        log(f"[WARN] Controls obtained: {got}/{need}", logpath)
        return 1

    log(f"[OK] Controls obtained: {got}/{need}", logpath)
    return 0


# -------------------------
# RECENT monitor
# -------------------------
def is_recent_event(event_rel: str) -> bool:
    return "RECENT" in str(event_rel).upper()

def event_tag_from_rel(event_rel: str) -> str:
    return str(event_rel).replace("/", "_").replace("\\", "_")

def project_root_from_core(core_dir: Path) -> Path:
    return core_dir.resolve().parents[1]

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the project root folder reliably.

    We prefer anchoring from this script's location (Path(__file__)) instead of the
    current working directory, because the pipeline is often executed from nested
    output folders (e.g., resultados/_summary/...).

    A folder is considered the project root if it contains:
      - src/core/  (recommended layout)
        OR
      - src/       (fallback)
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    start_path = Path(start_path).resolve()
    # If a file path is provided, start searching from its parent directory.
    if start_path.is_file():
        cur = start_path.parent
    else:
        cur = start_path

    for p in [cur] + list(cur.parents):
        if (p / "src" / "core").exists():
            return p
        if (p / "src").exists():
            return p

    # Last resort: fall back to the current working directory.
    return Path.cwd().resolve()
def build_recent_summary_png(event_rel: str, core_dir: Path) -> Optional[Path]:
    pr = project_root_from_core(core_dir)
    tag = event_tag_from_rel(event_rel)
    plots_dir = pr / "resultados" / Path(event_rel) / "plots"

    f_forz = plots_dir / f"forzantes_{tag}_24h.png"
    f_rob  = plots_dir / f"robust_precursors_{tag}.png"

    if not f_forz.exists() or not f_rob.exists():
        print(f"[RECENT] No puedo crear resumen: faltan archivos en {plots_dir}")
        print(f"         - {f_forz.name} existe? {f_forz.exists()}")
        print(f"         - {f_rob.name} existe? {f_rob.exists()}")
        return None

    try:
        img1 = mpimg.imread(str(f_forz))
        img2 = mpimg.imread(str(f_rob))
    except Exception as e:
        print(f"[RECENT] Error leyendo PNGs para resumen: {repr(e)}")
        return None

    out_png = plots_dir / f"monitoring_summary_{tag}.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(img1); axes[0].axis("off"); axes[0].set_title("Forzantes / marea (24h)")
    axes[1].imshow(img2); axes[1].axis("off"); axes[1].set_title("Robust precursors (24h)")
    fig.suptitle(f"Monitorización visual (RECENT) — {event_rel}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


# =========================
# NO OPEN PNGs (FIX)
# =========================
def open_png(path: Path) -> None:
    """
    ES: NO abre PNG en pantalla. Solo informa ubicación.
    EN: Does NOT open PNG on screen. Only prints location.
    """
    print(f"[INFO] PNG guardado / PNG saved: {path}")

def open_all_plots(event_rel: str, core_dir: Path) -> None:
    """
    ES: NO abre nada. Solo informa dónde están los plots.
    EN: Does NOT open anything. Only reports plots directory.
    """
    pr = project_root_from_core(core_dir)
    plots_dir = pr / "resultados" / Path(event_rel) / "plots"
    if plots_dir.exists():
        print(f"[INFO] Plots guardados en / Plots saved in: {plots_dir}")


def run_recent_monitor(name: str, lat: float, lon: float) -> int:
    print("Analiza las correlaciones de las estaciones")
    print("¡NO ES UN SISTEMA DE PREDICCIÓN DE TERREMOTOS!")
    print("[MONITOR] Solo descarga la ventana RECENT (24h) y NO descarga días de control.")

    now = UTCDateTime()
    end = now - 3 * 3600
    start = end - 24 * 3600

    print(f"[MONITOR] Window: {start.isoformat()}Z -> {end.isoformat()}Z")

    base_dir = get_base_dir()
    ensure_dir(base_dir)

    stamp = end.strftime("%Y%m%d_%H%M%S")
    root_name = re_safe(f"{name}_RECENT_{stamp}")
    root_dir = os.path.join(base_dir, root_name)
    ensure_dir(root_dir)

    mainshock_dir = os.path.join(root_dir, "mainshock")
    mainshock_raw = os.path.join(mainshock_dir, "raw")
    ensure_dir(mainshock_raw)

    logpath = os.path.join(root_dir, "runlog.txt")
    with open(logpath, "w", encoding="utf-8") as f:
        f.write("RUNLOG\n")
        f.write("MODE=RECENT_MONITOR\n")
        f.write(f"NAME={name}\n")
        f.write(f"LAT={lat}\nLON={lon}\n")
        f.write(f"WINDOW_START={start}\nWINDOW_END={end}\n")

    try:
        station_client = Client(CFG["station_provider"])
        data_client = build_data_client()
    except Exception as e:
        print(f"[RECENT] Error creando clientes FDSN: {repr(e)}")
        return 2

    cands = fetch_candidate_stations(station_client, start, end, float(lat), float(lon))
    if not cands:
        log("[RECENT] No candidate stations found.", logpath)
        return 2

    nearest_ok = []
    for sta in cands:
        if len(nearest_ok) >= int(CFG["n_nearest"]):
            break
        ok = download_station_window(data_client, start, end, sta, mainshock_raw, logpath, tag="RECENT")
        if ok:
            nearest_ok.append(sta)

    if len(nearest_ok) < int(CFG["n_nearest"]):
        log(f"[RECENT] Only {len(nearest_ok)}/{CFG['n_nearest']} stations passed QC. Aborting.", logpath)
        return 2

    fixed_path = os.path.join(root_dir, "fixed_stations.json")
    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump([sta.__dict__ for sta in nearest_ok], f, indent=2, ensure_ascii=False)
    log(f"[RECENT] Fixed stations saved: {fixed_path}", logpath)

    pr = find_project_root()
    core_dir = pr / "src" / "core"
    event_rel = f"{root_name}/mainshock"
    rc = run_core_pipeline_for_event(event_rel, core_dir)
    return rc


# -------------------------
# Processing (00..07)
# -------------------------
def discover_subevents(event_dir: Path) -> List[str]:
    """
    ES: Devuelve subeventos procesables dentro de data/<evento>:
        - mainshock
        - control_*
    EN: Returns processable subevents inside data/<event>:
        - mainshock
        - control_*
    """
    subs = []
    for d in sorted(event_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "mainshock" or d.name.startswith("control_"):
            if (d / "raw").exists() or (d / "preprocessed" / "mseed_24h").exists():
                subs.append(d.name)
    return subs

def run_core_pipeline_for_event(event_rel: str, core_dir: Path) -> int:
    steps = [
        "00_preprocess_24h.py",
        "01_rotation_3c.py",
        "02_tamc_metrics.py",
        "03_scan_criticality.py",
        "04_sync_multistation.py",
        "05_mareas_forzantes.py",
        "06_plotting.py",
        "07_robust_precursors_single.py",
    ]

    print("\n" + "=" * 70)
    print(f"[PROCESS / PROCESAR] EVENT = {event_rel}")
    print("=" * 70)

    for s in steps:
        script_path = core_dir / s
        if not script_path.exists():
            print(f"[ERROR] No existe / Missing: {script_path}")
            return 2

        cmd = [sys.executable, str(script_path), event_rel]
        print(f"\n>> {s}\n   {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(core_dir))
        if r.returncode != 0:
            print(f"[ERROR] Falló / Failed: {s} (rc={r.returncode}) for {event_rel}")
            return int(r.returncode)

    print(f"\n[OK] Pipeline completo / Pipeline completed for {event_rel}")

    # ES/EN: NO abrir ventanas; solo reportar ubicación de plots.
    if is_recent_event(event_rel):
        out_png = build_recent_summary_png(event_rel, core_dir)
        if out_png is not None:
            print(f"[RECENT] Resumen guardado / Summary saved: {out_png}")
            if AUTO_OPEN_PLOTS:
                open_png(out_png)
            else:
                # no abrir
                pass
    else:
        if AUTO_OPEN_PLOTS:
            open_all_plots(event_rel, core_dir)
        else:
            # no abrir
            pr = project_root_from_core(core_dir)
            plots_dir = pr / "resultados" / Path(event_rel) / "plots"
            if plots_dir.exists():
                print(f"[INFO] Plots guardados en / Plots saved in: {plots_dir}")

    return 0


def list_event_folders(base_dir: Path) -> List[Path]:
    if not base_dir.is_dir():
        return []
    return [d for d in sorted(base_dir.iterdir()) if d.is_dir()]

def process_menu() -> int:
    base_dir = Path(get_base_dir())
    events = list_event_folders(base_dir)

    if not events:
        print(f"No hay carpetas de eventos / No event folders in: {base_dir}")
        return 2

    print("\n=== Eventos disponibles en data/ / Available events in data/ ===\n")
    for i, d in enumerate(events, 1):
        print(f"{i:02d}) {d.name}")

    raw = input("\nSeleccioná evento(s) / Select event(s) (ej: '1' o '1 3 5'):\n> ").strip()
    idxs = parse_indices(raw, len(events))
    if not idxs:
        print("Selección inválida / Invalid selection.")
        return 2

    project_root = find_project_root()
    core_dir = project_root / "src" / "core"
    if not core_dir.is_dir():
        print(f"[ERROR] No encuentro src/core en / Missing src/core in: {project_root}")
        return 2

    rc_max = 0
    for k in idxs:
        event_dir = events[k - 1]
        event_name = event_dir.name

        subs = discover_subevents(event_dir)
        if subs:
            ordered = (["mainshock"] if "mainshock" in subs else []) + [s for s in subs if s != "mainshock"]
            for sub in ordered:
                event_rel = f"{event_name}/{sub}"
                rc = run_core_pipeline_for_event(event_rel, core_dir)
                rc_max = max(rc_max, rc)
        else:
            rc = run_core_pipeline_for_event(event_name, core_dir)
            rc_max = max(rc_max, rc)

    return rc_max


# -------------------------
# Download menu
# -------------------------
def download_menu() -> int:
    feats = fetch_top_earthquakes_last_20y(
        limit=int(CFG["usgs_limit"]),
        minmag=float(CFG["usgs_minmag"])
    )
    tor = {"name": "Torremolinos (Málaga), Spain", "lat": 36.6215, "lon": -4.4994}

    while True:
        print("\n=== TERREMOTOS / EARTHQUAKES ===\n")

        print("01) [PRESET] Japan 2026-04-20 07:53 UTC — M7.4 near Miyako (USGS id=us6000sri7)")
        print("02) [MONITOR] Torremolinos (Málaga), Spain  (always available)")

        for i, feat in enumerate(feats, 3):
            print(pretty_event_row(i, feat))

        raw = input("Elige índice(s) / Choose index(es) (ej: '1' o '3 5 8') | 'q' salir/quit:\n> ").strip().lower()

        if raw in {"q", "quit", "salir", "exit"}:
            return 0

        idxs = parse_indices(raw, len(feats) + 2)
        if not idxs:
            print("Selección inválida / Invalid selection.")
            continue

        rc_max = 0
        for idx in idxs:
            if idx == 1:
                feat = fetch_usgs_event_by_id("us6000sri7")
                if feat is None:
                    print("[ERROR] No pude obtener el evento de Japón reciente (us6000sri7).")
                    rc_max = max(rc_max, 2)
                    continue
            elif idx == 2:
                rc = run_recent_monitor(tor["name"], tor["lat"], tor["lon"])
                rc_max = max(rc_max, rc)
                continue
            else:
                feat = feats[idx - 3]

            p = feat["properties"]
            place = p.get("place", "Unknown place")

            print(f"=== Selección / Selected: {place} ===")
            print("¿Qué quieres hacer? / What do you want to do?")
            print("  [E] Descargar datos + controles / Download event + controls")
            print("  [M] Monitorizar 24h (SIN controles) / Monitor last 24h (NO controls)")
            mode = input("Elige [E/M] (E=download, M=monitor): ").strip().lower() or "e"

            if mode in {"m", "monitor"}:
                rc = run_recent_monitor(place, 0.0, 0.0)  # kept structure
                rc_max = max(rc_max, rc)
            else:
                default_cd = int(CFG["control_days"])
                raw_cd = input(f"¿Cuántos controles? / How many control days? (Enter = {default_cd}):\n> ").strip()

                if raw_cd == "":
                    cd = default_cd
                elif raw_cd.isdigit() and int(raw_cd) >= 0:
                    cd = int(raw_cd)
                else:
                    print("Valor inválido. Uso default. / Invalid value, using default.")
                    cd = default_cd

                rc = run_event_download(feat, control_days=cd)
                rc_max = max(rc_max, rc)

        return rc_max


# -------------------------
# NULLTESTS suite: uses 11 + 10 (09 eliminado)
# -------------------------
def run_tests_11_10(events: List[str]) -> int:
    if not events:
        return 0

    project_root = find_project_root()
    core_dir = project_root / "src" / "core"
    if not core_dir.is_dir():
        print(f"[ERROR] No encuentro src/core en / Missing src/core in: {project_root}")
        return 2

    py = sys.executable

    def run_core_cmd(cmd: List[str]) -> int:
        print("\n" + "=" * 90)
        print("COMANDO / COMMAND:", " ".join(cmd))
        print("=" * 90)
        r = subprocess.run(cmd, cwd=str(core_dir))
        return int(r.returncode)

    for ev in events:
        rc = run_core_cmd([py, "11_null_event_vs_controls.py", "--event", ev, "--no-show"])
        if rc != 0:
            return rc

        ensure_event_root_metrics_for_placebo(ev)

        event_date = infer_event_date_from_name(ev)
        cmd10 = [py, "10_placebo_matched_controls_FINAL.py", "--event", ev]
        if event_date:
            cmd10 += ["--event-date", event_date]
        else:
            print(f"[WARN] No pude inferir fecha desde el nombre '{ev}'.")

        rc = run_core_cmd(cmd10)
        if rc != 0:
            return rc

        # --- 09 (block shuffle null) ---
        script09 = core_dir / "09_null_block_shuffle.py"
        if script09.exists():
            # Guardar dentro del evento: resultados/<EVENTO>/nulltest/09_block_shuffle/
            pr = find_project_root()
            ev_dir = pr / "resultados" / Path(ev)
            out09 = ev_dir / "nulltest" / "09_block_shuffle"
            rc = run_core_cmd([py, "09_null_block_shuffle.py", "--event", ev, "--out-dir", str(out09)])
            if rc != 0:
                return rc
        else:
            print(f"[WARN] Falta {script09}. Se omite el paso 09 (block shuffle).")

    # 09 eliminado (no aporta y a veces no existe en src/core)
    return 0

def run_tests_11_10_09(events: List[str]) -> int:
    """Compat: nombre antiguo. Ejecuta 11 + 10 (09 eliminado)."""
    return run_tests_11_10(events)


# -------------------------
# Helpers: reporting controls in a dir
# -------------------------
def count_control_days_per_event(dir_path: str) -> List[Tuple[str, int]]:
    base = Path(dir_path)
    if not base.is_dir():
        return []
    out: List[Tuple[str, int]] = []
    for ev in sorted([d for d in base.iterdir() if d.is_dir()]):
        n = 0
        try:
            for sub in ev.iterdir():
                if sub.is_dir() and sub.name.startswith("control_"):
                    n += 1
        except Exception:
            pass
        out.append((ev.name, n))
    return out

def print_control_days_report(dir_path: str, title: str = "") -> None:
    rows = count_control_days_per_event(dir_path)
    if title:
        print("\n" + title)
    print(f"\n=== DÍAS DE CONTROL / CONTROL DAYS en: {dir_path} ===")
    if not rows:
        print("(vacío / empty)")
        return
    for name, n in rows:
        print(f"- {name}  ->  {n} controles / controls")


# -------------------------
# RESULTADOS: find valid events, including resultados/otros/*
# -------------------------
def _count_controls_inside_event_folder(ev_dir: Path) -> int:
    n = 0
    try:
        for sub in ev_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith("control_"):
                n += 1
    except Exception:
        pass
    return n

def find_valid_resultados_events(min_controls: int = 5) -> List[str]:
    pr = find_project_root()
    resultados_dir = pr / "resultados"
    if not resultados_dir.is_dir():
        return []

    valid: List[str] = []

    for ev_dir in resultados_dir.iterdir():
        if not ev_dir.is_dir():
            continue
        ev = ev_dir.name
        low = ev.lower()
        if low.startswith(("control_", "multi_", "null_tests")):
            continue
        if (ev_dir / "mainshock").is_dir():
            n_controls = _count_controls_inside_event_folder(ev_dir)
            if n_controls >= int(min_controls):
                valid.append(ev)

    otros_dir = resultados_dir / "otros"
    if otros_dir.is_dir():
        for ev_dir in otros_dir.iterdir():
            if not ev_dir.is_dir():
                continue
            if not (ev_dir / "mainshock").is_dir():
                continue
            n_controls = _count_controls_inside_event_folder(ev_dir)
            if n_controls >= int(min_controls):
                valid.append(f"otros/{ev_dir.name}")

    return sorted(valid)

def runtests_resultados_menu() -> int:
    valid = find_valid_resultados_events(min_controls=5)
    if not valid:
        print("\n[ERROR] No encontré eventos válidos en resultados/.")
        print("Requisito: resultados/<EVENTO>/mainshock/ (o resultados/otros/<EVENTO>/mainshock/) y >=5 controles.\n")
        return 2

    print("\n=== RUNTESTS en resultados/ / RunTests in resultados/ ===\n")
    for i, ev in enumerate(valid, 1):
        print(f"{i:02d}) {ev}")

    raw = input("\nElige índice(s) / Choose index(es) (ej: '1' o '1 3 5') | 'b' volver/back:\n> ").strip().lower()
    if raw in {"b", "back", "volver", ""}:
        return 0

    idxs = parse_indices(raw, len(valid))
    if not idxs:
        print("Selección inválida / Invalid selection.")
        return 2

    chosen = [valid[i - 1] for i in idxs]
    return run_tests_11_10(chosen)


# -------------------------
# Nulltest menu (coherent)
# -------------------------
def nulltest_menu() -> int:
    base_dir = get_base_dir()
    null_dir = os.path.join(base_dir, "otros")
    ensure_dir(null_dir)

    print_control_days_report(null_dir, title="(Antes de descargar / Before download)")

    feats = fetch_top_earthquakes_last_20y(limit=int(CFG["usgs_limit"]), minmag=float(CFG["usgs_minmag"]))

    while True:
        print(r"""
=== NULL TEST (coherente / coherent) ===
Selecciona eventos USGS. Se guardarán en / Saved in: data/otros/
""")
        for i, feat in enumerate(feats, 1):
            print(pretty_event_row(i, feat))

        raw = input("Elige índice(s) / Choose index(es) (ej: '1' o '1 3 5') | 'b' volver/back | 'q' salir/quit:\n> ").strip().lower()
        if raw in {"q", "quit", "salir", "exit"}:
            return 0
        if raw in {"b", "back", "volver"}:
            return 0

        idxs = parse_indices(raw, len(feats))
        if not idxs:
            print("Selección inválida / Invalid selection.")
            continue

        before = {d.name for d in Path(null_dir).iterdir() if d.is_dir()}

        rc_max = 0
        for idx in idxs:
            feat = feats[idx - 1]
            place = feat.get("properties", {}).get("place", "Unknown place")
            print(f"\n=== NULL TEST: {place} ===")

            default_cd = int(CFG["control_days"])
            raw_cd = input(f"¿Cuántos controles? / How many controls? (Enter = {default_cd}):\n> ").strip()
            if raw_cd == "":
                cd = default_cd
            elif raw_cd.isdigit() and int(raw_cd) >= 0:
                cd = int(raw_cd)
            else:
                print("Valor inválido. Uso default. / Invalid value, using default.")
                cd = default_cd

            rc = run_event_download(feat, control_days=cd, base_dir_override=null_dir)
            rc_max = max(rc_max, rc)

        print_control_days_report(null_dir, title="(Después de descargar / After download)")

        after = {d.name for d in Path(null_dir).iterdir() if d.is_dir()}
        created = sorted(list(after - before))

        if created:
            print("\n=== Eventos NULL descargados / Downloaded NULL events in data/otros/ ===")
            for name in created:
                print(f"- {name}")
        else:
            print("\n[WARN] No detecté nuevas carpetas creadas / No new folders detected in data/otros/.")
            return rc_max

        do_process = input("\n¿Procesar ahora (00..07) estos eventos? / Process now (00..07)? [S/n]:\n> ").strip().lower()
        if do_process in {"", "s", "si", "sí", "y", "yes"}:
            pr = find_project_root()
            core_dir = pr / "src" / "core"
            if not core_dir.is_dir():
                print(f"[ERROR] No encuentro src/core en / Missing src/core in: {pr}")
                return 2

            for ev_name in created:
                ev_dir = Path(null_dir) / ev_name
                subs = discover_subevents(ev_dir)
                ordered = (["mainshock"] if "mainshock" in subs else []) + [s for s in subs if s != "mainshock"]
                for sub in ordered:
                    event_rel = f"otros/{ev_name}/{sub}"
                    rc = run_core_pipeline_for_event(event_rel, core_dir)
                    rc_max = max(rc_max, rc)

        do_tests = input("\n¿Correr ahora RunTests (11/10/09) sobre resultados/otros/*? / Run RunTests now? [S/n]:\n> ").strip().lower()
        if do_tests in {"", "s", "si", "sí", "y", "yes"}:
            ev_keys = [f"otros/{name}" for name in created]
            rc_tests = run_tests_11_10(ev_keys)
            rc_max = max(rc_max, rc_tests)

        return rc_max



# -------------------------
# Inter-event analysis (13..18)
# -------------------------
INTER_EVENT_SCRIPTS: List[str] = [
    # Inter-event REAL (mainshock vs mainshock). No event-vs-controls (12) here.
    "13_compare_events_phases.py",
    "14_station_coverage_phases.py",
    "15_null_tests_13_14.py",
    "15_null_viz_pvalues_13_14.py",
    "16_make_pvalues_table_for_paper.py",
    "17_nullA_and_C_robustness.py",
    "18_extract_final_pvalues.py",
]

def list_resultados_event_folders() -> List[str]:
    """
    ES: Lista carpetas de eventos dentro de resultados/ y resultados/otros/.
    EN: Lists event folders inside resultados/ and resultados/otros/.
    Devuelve rutas relativas tipo:
      - "<EVENTO>"
      - "otros/<EVENTO>"
    """
    pr = find_project_root()
    resultados_dir = pr / "resultados"
    if not resultados_dir.is_dir():
        return []

    out: List[str] = []

    # root resultados/
    for ev_dir in sorted([d for d in resultados_dir.iterdir() if d.is_dir()]):
        name = ev_dir.name
        if name == "otros":
            continue
        # Excluir carpetas no-evento
        if name.endswith("_nulltest") or name in {"multi_nulltest", "resultados"}:
            continue
        # Solo eventos con métricas reales
        metrics_csv = ev_dir / "metrics" / "tamc_24h_metrics_allstations.csv"
        if not metrics_csv.is_file():
            continue
        out.append(name)

    # resultados/otros/
    otros_dir = resultados_dir / "otros"
    if otros_dir.is_dir():
        for ev_dir in sorted([d for d in otros_dir.iterdir() if d.is_dir()]):
            name = ev_dir.name
            if name.endswith("_nulltest") or name in {"multi_nulltest", "resultados"}:
                continue
            metrics_csv = ev_dir / "metrics" / "tamc_24h_metrics_allstations.csv"
            if not metrics_csv.is_file():
                continue
            out.append(f"otros/{name}")

    return out


def choose_up_to_4_events_from_resultados() -> List[str]:
    """
    ES: Permite elegir hasta 4 eventos de resultados/.
    EN: Lets you choose up to 4 events from resultados/.
    """
    events = list_resultados_event_folders()
    if not events:
        print("[ERROR] No encontré carpetas en resultados/ (ni resultados/otros/).")
        return []

    print("\n=== EVENTOS DISPONIBLES EN resultados/ / AVAILABLE EVENTS IN resultados/ ===\n")
    for i, ev in enumerate(events, 1):
        print(f"{i:02d}) {ev}")

    raw = input("\nElige hasta 4 índices / Choose up to 4 indices (ej: '1 3 5 7') | 'b' volver/back:\n> ").strip().lower()
    if raw in {"b", "back", "volver"}:
        return []

    idxs = parse_indices(raw, len(events))
    if not idxs:
        print("Selección inválida / Invalid selection.")
        return []
    if len(idxs) > 4:
        print("[ERROR] Máximo 4 eventos / Max 4 events.")
        return []

    chosen = [events[i - 1] for i in idxs]
    print("Seleccionados / Selected:")
    for ev in chosen:
        print(f"- {ev}")
    return chosen


def run_inter_event_analysis(events: List[str]) -> int:
    """
    ES: Ejecuta scripts de análisis inter-eventos REAL (13..18) en src/core.
    EN: Runs REAL inter-event analysis scripts (13..18) inside src/core.

    IMPORTANT:
      - No ejecuta el 12 (event vs controls); solo 13..18.
      - Ya seleccionamos eventos (máx 4) desde resultados/.
      - Por compatibilidad:
          * Intenta correr cada script con:  --root <project_root>  + events
          * Si falla por '--root' no reconocido -> reintenta sin --root
          * Si sigue fallando por argumentos (p.ej. el 12 no acepta eventos) -> reintenta sin eventos
    """
    if not events:
        return 0

    project_root = find_project_root()
    core_dir = project_root / "src" / "core"
    if not core_dir.is_dir():
        print(f"[ERROR] No encuentro src/core en / Missing src/core in: {project_root}")
        return 2

    resultados_dir = project_root / "resultados"
    py = sys.executable

    def run_cmd(script_name: str) -> int:
        script_path = core_dir / script_name
        if not script_path.exists():
            print(f"[WARN] Falta / Missing: {script_path} (se omite / skipping)")
            return 0

        def _print_header(cmd: List[str]) -> None:
            print("\n" + "=" * 90)
            print("ANÁLISIS INTER-EVENTOS / INTER-EVENT ANALYSIS")
            print("COMANDO / COMMAND:", " ".join(cmd))
            print("=" * 90)

        # Try 1: with --root + events
        cmd1 = [py, script_name, "--root", str(project_root)] + events
        _print_header(cmd1)
        p1 = subprocess.run(cmd1, cwd=str(core_dir), text=True, capture_output=True)

        if p1.returncode == 0:
            if p1.stdout:
                print(p1.stdout)
            return 0

        err1 = (p1.stderr or "") + "\n" + (p1.stdout or "")

        # Try 2: if --root not supported, retry without --root (keep events)
        if ("unrecognized arguments: --root" in err1) or ("--root" in (p1.stderr or "") and "unrecognized arguments" in (p1.stderr or "")):
            print("[WARN] '--root' no reconocido. Reintentando sin --root...")
            cmd2 = [py, script_name] + events
            print("COMANDO / COMMAND:", " ".join(cmd2))
            p2 = subprocess.run(cmd2, cwd=str(core_dir), text=True, capture_output=True)

            if p2.returncode == 0:
                if p2.stdout:
                    print(p2.stdout)
                return 0

            err2 = (p2.stderr or "") + "\n" + (p2.stdout or "")

            # Try 3: if script doesn't accept positional events, retry without events
            if ("unrecognized arguments" in err2) or ("the following arguments are required" in err2):
                print("[WARN] Argumentos no compatibles. Reintentando sin eventos...")
                cmd3 = [py, script_name]
                print("COMANDO / COMMAND:", " ".join(cmd3))
                p3 = subprocess.run(cmd3, cwd=str(core_dir), text=True, capture_output=True)
                if p3.stdout:
                    print(p3.stdout)
                if p3.stderr:
                    print(p3.stderr)
                return int(p3.returncode)

            if p2.stdout:
                print(p2.stdout)
            if p2.stderr:
                print(p2.stderr)
            return int(p2.returncode)

        # Try 3: script supports --root but not events -> retry with --root only
        if ("unrecognized arguments" in err1) or ("the following arguments are required" in err1):
            print("[WARN] Argumentos no compatibles. Reintentando sin eventos...")
            cmd3 = [py, script_name, "--root", str(project_root)]
            print("COMANDO / COMMAND:", " ".join(cmd3))
            p3 = subprocess.run(cmd3, cwd=str(core_dir), text=True, capture_output=True)
            if p3.returncode == 0:
                if p3.stdout:
                    print(p3.stdout)
                return 0

            # Last resort: run script with no args
            print("[WARN] Reintentando solo script (sin args)...")
            cmd4 = [py, script_name]
            print("COMANDO / COMMAND:", " ".join(cmd4))
            p4 = subprocess.run(cmd4, cwd=str(core_dir), text=True, capture_output=True)
            if p4.stdout:
                print(p4.stdout)
            if p4.stderr:
                print(p4.stderr)
            return int(p4.returncode)

        # Default: show output and return rc
        if p1.stdout:
            print(p1.stdout)
        if p1.stderr:
            print(p1.stderr)
        return int(p1.returncode)

    # Run in fixed order (12..18)
    for s in INTER_EVENT_SCRIPTS:
        rc = run_cmd(s)
        if rc != 0:
            print(f"[ERROR] Falló / Failed: {s} (rc={rc})")
            return rc

    print("[OK] Análisis inter-eventos completado / Inter-event analysis completed.")
    return 0


def inter_event_menu() -> int:
    """
    ES: Opción 4: elegir hasta 4 eventos en resultados/ y correr 13..18 (Inter-event REAL).
    EN: Option 4: pick up to 4 events in resultados/ and run 13..18 (Inter-event REAL).
    """
    print("\n=== ANÁLISIS INTER-EVENTOS / INTER-EVENT ANALYSIS ===\n")
    print("Selecciona hasta 4 eventos desde resultados/ para comparar.")
    print("Select up to 4 events from resultados/ to compare.\n")

    chosen_events = choose_up_to_4_events_from_resultados()
    if not chosen_events:
        return 0

    return run_inter_event_analysis(chosen_events)

# -------------------------
# Main menu
# -------------------------

def find_script_19(project_root: Path) -> Optional[Path]:
    """Busca únicamente el clustering REAL (19_cluster_events.py o cluster_events.py).

    ES:
      - Prioriza scripts que usan features reales inter-evento (real_multi_phase_summary.csv).
      - NO usa 19_classify_earthquakes.py (ELI-based), para evitar problemas de matching
        de nombres y NaNs silenciosos.

    EN:
      - Prioritizes REAL inter-event clustering scripts.
      - Does NOT use 19_classify_earthquakes.py.
    """
    candidates = [
        # layout clásico
        project_root / "src" / "core" / "19_cluster_events.py",
        project_root / "src" / "core" / "cluster_events.py",
        # layout alternativo (src/)
        project_root / "src" / "19_cluster_events.py",
        project_root / "src" / "cluster_events.py",
        # root
        project_root / "19_cluster_events.py",
        project_root / "cluster_events.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def run_event_clustering(chosen_events: List[str]) -> int:
    """
    Ejecuta el script 19 (clasificación) sobre la lista de eventos.
    Pasa --root <project_root> y los eventos (si el 19 los soporta).
    """
    project_root = find_project_root()
    script19 = find_script_19(project_root)
    if script19 is None:
        print(f"[ERROR] No encuentro los scripts de clustering (19_cluster_events.py / cluster_events.py) en las rutas esperadas.")
        print(f"        Busqué en: {project_root / 'src' / 'core'}, {project_root / 'src'} y {project_root}.")
        print(f"[EN] Cannot find clustering scripts (19_cluster_events.py / cluster_events.py) in expected paths.")
        print(f"     Searched: {project_root / 'src' / 'core'}, {project_root / 'src'} and {project_root}.")
        return 2

    py = sys.executable
    # Intentamos con eventos; si el script no acepta, reintentamos sin lista
    cmd = [py, str(script19), "--root", str(project_root)] + chosen_events
    print("\n" + "="*90)
    print("CLASIFICACIÓN DE TERREMOTOS / EARTHQUAKE CLUSTERING")
    print("SCRIPT / SCRIPT:", script19)
    print("COMANDO / COMMAND:", " ".join(cmd))
    print("="*90)
    rc = run_cmd(cmd)
    if rc == 0:
        return 0
    if rc != 0:
        print("[ERROR] El clustering falló. No se reintenta sin eventos (mantengo la selección del menú).")
        return rc




def eli_menu() -> int:
    print("\n=== CLASIFICACIÓN POR ELI / EXPLOSION-LIKENESS INDEX (ELI) ===\n")
    print("ES: Ejecuta 21_explosion_index.py y 22_make_eli_table_for_paper.py en src/core.")
    print("EN: Runs 21_explosion_index.py and 22_make_eli_table_for_paper.py from src/core.\n")

    core_dir = (Path(__file__).resolve().parent / "src" / "core")
    if not core_dir.is_dir():
        print(f"[ERROR] No existe el directorio: {core_dir}")
        print("[EN] src/core directory not found.")
        return 2

    s1 = core_dir / "21_explosion_index.py"
    s2 = core_dir / "22_make_eli_table_for_paper.py"
    missing = [str(p) for p in (s1, s2) if not p.exists()]
    if missing:
        print("[ERROR] Faltan scripts necesarios:")
        for p in missing:
            print("  -", p)
        print("[EN] Missing required scripts.")
        return 2

    rc = run_cmd([sys.executable, str(s1.name)], cwd=str(core_dir))
    if rc != 0:
        print("[ERROR] Falló 21_explosion_index.py; no se ejecuta el paso 22.")
        return rc

    rc = run_cmd([sys.executable, str(s2.name)], cwd=str(core_dir))
    return rc

def clustering_menu() -> int:
    print("\n=== CLASIFICACIÓN DE TERREMOTOS / EARTHQUAKE CLUSTERING ===\n")
    print("ES: Elige eventos (o 'A' para todos). Se generará resultados/earthquake_clustering/")
    print("EN: Choose events (or 'A' for all). Will write resultados/earthquake_clustering/\n")

    # Reutiliza el listado que ya filtra _nulltest y exige metrics reales
    events = list_resultados_event_folders()
    if not events:
        print("[ERROR] No hay eventos válidos en resultados/ (faltan metrics/tamc_24h_metrics_allstations.csv)")
        print("[EN] No valid events in resultados/ (missing metrics/tamc_24h_metrics_allstations.csv)")
        return 2

    print("\n=== EVENTOS DISPONIBLES EN resultados/ / AVAILABLE EVENTS IN resultados/ ===\n")
    for i, ev in enumerate(events, start=1):
        print(f"{i:02d}) {ev}")
    print("\nElige 'A' (todos/all) o índices (ej: '1 3 5') | 'b' volver/back:")

    raw = input("> ").strip()
    if raw.lower() in {"b", "back"}:
        return 0

    chosen: List[str] = []
    if raw.upper() == "A":
        chosen = events[:]
    else:
        idxs = parse_indices(raw, len(events))
        if not idxs:
            print("Entrada inválida / Invalid input.")
            return 2
        for ix in idxs:
            if 1 <= ix <= len(events):
                chosen.append(events[ix-1])

    if not chosen:
        print("Nada seleccionado / Nothing selected.")
        return 2

    print("\nSeleccionados / Selected:")
    for ev in chosen:
        print("-", ev)

    # Pregunta si queremos correr antes inter-event 13..18 (opcional)
    ans = input("\n¿Correr inter-event REAL (13..18) antes de clasificar? [s/N]: ").strip().lower()
    if ans in {"s", "si", "sí", "y", "yes"}:
        print("\n[INFO] Ejecutando inter-event (13..18) antes de la clasificación...")
        rc = run_inter_event_analysis(chosen)
        if rc != 0:
            print("[ERROR] Inter-event falló. No se ejecuta clasificación.")
            return rc

    return run_event_clustering(chosen)

def main() -> int:
    print("\n==============================")
    print("  TAMC MASTER MENU (ES/EN)")
    print("==============================")
    print("1) Descargar / Monitorizar  (Download / Monitor)")
    print("2) Procesar data/ existentes (Process existing data/)")
    print("3) RunTests sobre resultados/ (incluye resultados/otros/*)")
    print("4) Análisis inter-eventos (13..18) / Inter-event analysis (13..18)")
    print("5) Clasificación de terremotos (19) / Earthquake clustering (19)")
    print("6) Clasificación por ELI (21+22) / Explosion-likeness index (ELI)")
    print("0) Salir / Exit")

    opt = input("\nOpción / Option:\n> ").strip()

    if opt == "1":
        return download_menu()
    if opt == "2":
        return process_menu()
    if opt == "3":
        return runtests_resultados_menu()
    if opt == "4":
        return inter_event_menu()
    if opt == "5":
        return clustering_menu()
    if opt == "6":
        return eli_menu()
    if opt == "0":
        return 0

    print("Opción inválida / Invalid option.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# regenerated link marker

# DATEFIX6: removed step 09 from RunTests

# regenerated DATEFIX7
# DATEFIX8_CLEAN: added step 09_null_block_shuffle.py after placebo (no stray \\n)